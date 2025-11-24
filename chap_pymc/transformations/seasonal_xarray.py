import pandas as pd
import pydantic
import xarray


def week_to_period_weights(week: int) -> list[tuple[int, float]]:
    """Map an ISO week (1-53) to normalized periods (0-51) with weights.

    Splits the year into 52 equal periods and calculates overlap weights.
    Each period represents 365.25/52 ≈ 7.0288 days.

    Args:
        week: ISO week number (1-53)

    Returns:
        List of (period_index, weight) tuples where weights sum to 1.0
    """
    days_per_period = 365.25 / 52

    # Week spans days [(week-1)*7, week*7) in 0-indexed days
    week_start_day = float((week - 1) * 7)
    week_end_day = float(week * 7)

    # Handle week 53: cap at 365.25 days
    week_end_day = min(week_end_day, 365.25)
    week_length = week_end_day - week_start_day

    if week_length <= 0:
        return [(51, 1.0)]  # Edge case: return last period

    weights: list[tuple[int, float]] = []

    for period in range(52):
        period_start = period * days_per_period
        period_end = (period + 1) * days_per_period

        # Calculate overlap between week and period
        overlap_start = max(week_start_day, period_start)
        overlap_end = min(week_end_day, period_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > 0:
            weight = overlap / week_length
            weights.append((period, weight))

    return weights


def normalize_weekly_data(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    """Normalize weekly data to 52 equal periods using weighted distribution.

    Args:
        df: DataFrame with 'location', 'year', 'week' (0-indexed), 'time_period', and value columns
        value_columns: Names of columns containing values to distribute

    Returns:
        DataFrame with 'week' replaced by normalized period (0-51), values are weighted averages
    """
    import numpy as np

    # Check input for NaNs
    for col in value_columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"normalize_weekly_data input: {nan_count} NaNs in {col}")

    rows = []

    for _, row in df.iterrows():
        week_1indexed = int(row['week']) + 1
        weights = week_to_period_weights(week_1indexed)

        for period, weight in weights:
            new_row = {
                'location': row['location'],
                'year': row['year'],
                'week': period,
                'time_period': row['time_period'],
            }
            # Track weights separately for each column, excluding NaN values
            for col in value_columns:
                val = row[col]
                if pd.isna(val):
                    new_row[col] = 0.0
                    new_row[f'_weight_{col}'] = 0.0
                else:
                    new_row[col] = val * weight
                    new_row[f'_weight_{col}'] = weight
            rows.append(new_row)

    result = pd.DataFrame(rows)

    # Aggregate by location, year, period - take first time_period for the mapping
    agg_dict: dict[str, str] = {}
    for col in value_columns:
        agg_dict[col] = 'sum'
        agg_dict[f'_weight_{col}'] = 'sum'
    agg_dict['time_period'] = 'first'  # Keep one time_period for coord mapping

    aggregated = result.groupby(['location', 'year', 'week']).agg(agg_dict).reset_index()

    # Convert weighted sums back to weighted averages
    # If total weight is 0, result is NaN (all inputs were NaN)
    for col in value_columns:
        weight_col = f'_weight_{col}'
        aggregated[col] = np.where(
            aggregated[weight_col] > 0,
            aggregated[col] / aggregated[weight_col],
            np.nan
        )
        aggregated = aggregated.drop(columns=[weight_col])

    # Check output for NaNs
    for col in value_columns:
        nan_count = aggregated[col].isna().sum()
        if nan_count > 0:
            print(f"normalize_weekly_data output: {nan_count} NaNs in {col}")

    return aggregated




class SeasonInformation:
    season_length: int
    name: str

    @classmethod
    def get(cls, frequency: str):
        if frequency == 'M':
            return MonthInfo()
        elif frequency == 'W':
            return WeekInfo()
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

class WeekInfo(SeasonInformation):
    def __init__(self):
        ...

    season_length: int = 52
    name: str = "week"

class MonthInfo(SeasonInformation):
    season_length: int = 12
    name: str = "month"

class TimeCoords(pydantic.BaseModel):
    epi_year: int
    epi_offset: int


class SeasonalXArray:
    class Params(pydantic.BaseModel):
        split_season_index: int | None = None
        target_variable: str = "disease_cases"
        frequency: str = "M"
        alignment: str = "min"  # 'min' or 'mid'

    def __init__(self, params: Params=Params()):
        self._params = params
        self._season_info = SeasonInformation.get(params.frequency)

    def create_coord_mapping(self, data_frame: pd.DataFrame) -> dict[str, TimeCoords]:
        return {str(row['time_period']): TimeCoords(epi_year=int(row['epi_year']), epi_offset=int(row['epi_offset'])) for _, row in data_frame.iterrows()}


    def get_dataset(self, data_frame: pd.DataFrame) -> tuple[xarray.Dataset, dict[str, TimeCoords]]:
        data_frame = data_frame.copy()

        # Parse time_period based on format
        # For weekly: can be "YYYY-WW" or "YYYY-MM-DD/YYYY-MM-DD" (date range)
        # For monthly: "YYYY-MM"
        def parse_period(time_period: str) -> tuple[int, int]:
            """Parse time_period and return (year, period_index).

            Supports:
            - Monthly: "2024-07" -> (2024, 7)
            - Weekly simple: "2024-15" -> (2024, 15)
            - Weekly date range: "2024-01-08/2024-01-14" -> (2024, week_in_year)
            """
            if '/' in time_period:
                # Date range format: "YYYY-MM-DD/YYYY-MM-DD"
                # Use start date to determine week of year
                start_date = pd.to_datetime(time_period.split('/')[0])
                year = start_date.year
                # Calculate week number based on calendar year (not ISO year)
                # Week 1 starts on Jan 1, regardless of day of week
                day_of_year = start_date.timetuple().tm_yday
                week = (day_of_year - 1) // 7 + 1  # 1-indexed, integer division
                return year, week
            else:
                # Simple format: "YYYY-PP" where PP is month or week number
                parts = time_period.split('-')
                year = int(parts[0])
                period = int(parts[1])
                return year, period

        periods = data_frame['time_period'].apply(parse_period)
        data_frame['year'] = periods.apply(lambda x: x[0])
        data_frame[self.freq_name] = periods.apply(lambda x: x[1] - 1)  # 0-indexed

        # Normalize weekly data to 52 equal periods
        if self._params.frequency == 'W':
            # Get value columns (numeric columns excluding year, week)
            value_columns = [col for col in data_frame.columns
                           if col not in ['location', 'year', 'week', 'time_period']
                           and pd.api.types.is_numeric_dtype(data_frame[col])]
            data_frame = normalize_weekly_data(data_frame, value_columns)

        self._min_month = self._find_min_month(data_frame) if self._params.split_season_index is None else self._params.split_season_index
        data_frame['epi_offset'] = (data_frame[self.freq_name] - self._min_month) % self.season_length
        offset = (data_frame[self.freq_name] - self._min_month) // self.season_length
        data_frame['epi_year'] = data_frame['year'] + offset
        # Set epi_year coords to count from -N to 0, where 0 is the last season
        data_frame['epi_year'] = data_frame['epi_year'] - data_frame['epi_year'].max()
        ds = xarray.Dataset.from_dataframe(data_frame.set_index(['location', 'epi_year', 'epi_offset']))
        return ds, self.create_coord_mapping(data_frame)

    @property
    def freq_name(self):
        return self._season_info.name

    @property
    def season_length(self) -> int:
        return int(self._season_info.season_length)

    def _find_min_month(self, data_frame: pd.DataFrame) -> int:
        means: list[tuple[int, float]] = [(month, group[self._params.target_variable].mean()) for month, group in data_frame.groupby(self.freq_name)]
        min_month, val  = min(means, key=lambda x: x[1])
        max_month, val = max(means, key=lambda x: x[1])

        if self._params.alignment == 'min':
            return min_month
        assert False, "Only 'min' alignment is currently supported"
        med: int = (min_month + max_month - 6) / 2  # type: ignore
        med = int(med - 1) % self.season_length + 1
        return med

