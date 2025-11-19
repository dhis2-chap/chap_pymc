import pandas as pd
import pydantic
import xarray




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

    season_length: int = 53
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
        return self._season_info.season_length

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

