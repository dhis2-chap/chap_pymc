# Jira CLI Commands Reference

Use the Atlassian CLI (acli) to interact with Jira. This project's main epic is **CLIM-140** (chap_pymc model).

## Authentication

Check if authenticated:
```bash
acli auth status
```

Login (opens browser for OAuth):
```bash
acli auth login
```

## Projects

List all projects:
```bash
acli jira project list --limit 20
```

## Work Items (Issues)

### Search Issues

Search with JQL query:
```bash
acli jira workitem search --jql "YOUR_JQL_QUERY" --limit 10
```

My assigned issues:
```bash
acli jira workitem search --jql "assignee = currentUser() ORDER BY updated DESC" --limit 10
```

Issues in Climate project:
```bash
acli jira workitem search --jql "project = CLIM ORDER BY updated DESC" --limit 10
```

### View Issue Details

```bash
acli jira workitem view ISSUE-KEY
```

Example:
```bash
acli jira workitem view CLIM-140
```

### Create Work Item

```bash
acli jira workitem create --project CLIM --type Task --summary "Summary text" --description "Detailed description"
```

Available types: Task, Story, Bug, Epic, etc.

### Transition Work Item

```bash
acli jira workitem transition --key ISSUE-KEY --status "Status Name" --yes
```

Common statuses: "To Do", "In Progress", "Done"

Example:
```bash
acli jira workitem transition --key CLIM-146 --status "Done" --yes
```

## Comments

Add a comment to an issue:
```bash
acli jira workitem comment create --key "ISSUE-KEY" --body "Your comment text"
```

List comments on an issue:
```bash
acli jira workitem comment list --key "ISSUE-KEY" --limit 5
```

## Useful JQL Patterns

- **Assigned to me**: `assignee = currentUser()`
- **Recent updates**: `ORDER BY updated DESC`
- **Specific project**: `project = PROJECTKEY`
- **Status filter**: `status = "In Progress"`
- **Multiple conditions**: `project = CLIM AND assignee = currentUser() AND status != Done`

## Common Workflows

### Check your tasks
```bash
acli jira workitem search --jql "assignee = currentUser() ORDER BY updated DESC" --limit 10
```

### Update issue with progress
```bash
acli jira workitem comment create --key "CLIM-140" --body "Progress update: [your update here]"
```

### View project issues
```bash
acli jira workitem search --jql "project = CLIM ORDER BY created DESC" --limit 20
```

## Getting Help

- General help: `acli --help`
- Jira commands: `acli jira --help`
- Workitem commands: `acli jira workitem --help`
- Specific command help: `acli jira workitem search --help`

## Project Context

**Main Epic**: CLIM-140 - chap_pymc model
- Goal: Publish a monthly version of the curve parametrization based pymc model
- Project: Climate (CLIM)
- Organization: DHIS2

### Active Tasks

**CLIM-141**: Implement proper NormalMixture model for seasonal patterns
- Replace current hacky weighted average with PyMC's NormalMixture
- Two components: (1) Empty season baseline, (2) Normal seasonal Fourier signal
- Each observation comes from ONE component (not weighted average)
- Proper statistical mixture model for climate-driven disease dynamics
- Related code: `fourier_parametrization.py:38-54` (_mixture_weights method)
- Status: To be implemented

## Additional Resources

- ACLI Documentation: https://developer.atlassian.com/cloud/acli/
- JQL Reference: https://support.atlassian.com/jira-software-cloud/docs/use-advanced-search-with-jira-query-language-jql/