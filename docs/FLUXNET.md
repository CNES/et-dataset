# FLUXNET Shuttle Library


A Python library for FLUXNET shuttle to discover and download global FLUXNET data from multiple data hubs, including AmeriFlux, ICOS, and TERN.

## Features
- **Data Download**: Download FLUXNET data from data hubs
- **Metadata Snapshot**: List metadata for FLUXNET data available via the data hubs
- **Command Line Interface**: Easy-to-use CLI tool `fluxnet-shuttle` for common operations
- **Comprehensive Logging**: Configurable logging with multiple outputs
- **Error Handling**: Custom exception handling for FLUXNET operations
- **Documentation and Examples**: Documentation for CLI and API (direct python usage) and example Jupyter Notebook

## Data Use Requirements

The FLUXNET data are shared under a [CC-BY-4.0 data use license](https://creativecommons.org/licenses/by/4.0/) which requires attribution for each data use.
See the data use license document contained within the FLUXNET data product (archive zip file) for details.

## Installation

## Command Line Interface (CLI)

The library includes a command-line tool `fluxnet-shuttle` that provides easy access to core functionality:

### CLI Commands

#### `listall`
Discover all available FLUXNET data products and their metadata:
```bash
fluxnet-shuttle --verbose listall
```
- Queries all connected data hubs
- Creates a timestamped CSV file with metadata and download information

#### `download`
Download data for specific sites:
```bash
# Download specific sites
fluxnet-shuttle download -f fluxnet_shuttle_snapshot_YYYYMMDDTHHMMSS.csv -s IT-Niv NZ-ADd

# Download ALL sites from snapshot (prompts for confirmation)
fluxnet-shuttle download -f fluxnet_shuttle_snapshot_YYYYMMDDTHHMMSS.csv

# Download ALL sites without confirmation prompt and skip prompts to enter optional user information
fluxnet-shuttle download -f fluxnet_shuttle_snapshot_YYYYMMDDTHHMMSS.csv --quiet
```
- Requires a CSV snapshot file from the `listall` command (`-f/--snapshot-file`)
- Specify site IDs with `-s/--sites` to download specific sites only
- Omit `-s/--sites` to download all sites in the snapshot (will prompt for confirmation unless `-q/--quiet` is used)
- The `-q/--quiet` flag skips prompts to enter optional user information and confirmation prompt when downloading all sites from a snapshot file.
- Downloads are saved to the output directory (default: current directory, use `-o` to specify)

### CLI Options
- `-v/--verbose`: Enable detailed logging output
- `-l/--logfile`: Specify log file path (default: `fluxnet-shuttle-run.log`)
- `--no-logfile`: Disable file logging, output only to console
- `--version`: Show version information
- `--help/-h`: Get help and see all options

### Example Workflow
```bash
# Step 1: Discover available data
fluxnet-shuttle --verbose listall

# Step 2: Download specific sites
fluxnet-shuttle --verbose download \
  -f fluxnet_shuttle_snapshot__20251006T155754.csv \
  -s NZ-ADd IT-Niv \

```
For more information : https://github.com/fluxnet/shuttle/blob/main/README.md
