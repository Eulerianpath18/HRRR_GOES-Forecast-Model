# Desktop forecasts

The GUI runs locally on the CPU. It can import a YAML configuration, choose a
weather station, download inputs for the latest eligible HRRR cycle, display the
satellite image, edit numerical inputs, and use a trained model for prediction.
The training archive stays on ARC. You download only the exported models.

## Open it

In PowerShell, from this project's folder with your existing environment:

```powershell
.\.venv\Scripts\python.exe forecast_gui.py
```

Alternatively, run `forecast_gui.py` in PyCharm using the project interpreter.
`launch_gui.bat` provides a double-click launcher when `.venv` is in this folder.
If extracting this release into a new directory, select the working interpreter
you already installed; do not create another environment unnecessarily. Install
the updated package into that interpreter once with `python -m pip install -e .`
if it still points to an older project directory. On Windows use that interpreter's
full path instead of `python` if Python is not on PATH.

**Before training:** click **View example** to inspect the included historical
inputs, change the location, and open **Satellite overview**. Click the image to
enlarge it. These are real historical inputs, not a trained forecast demonstration.
You can also download recent inputs before training. Predictions require trained
weights; this release does not include scientifically trained weights.

## Use a trained model

The ARC report job automatically exports the selected neural models into
`$FORECAST_RUNS/gui_models`. Each `.pt` bundle includes weights, training
normalization, feature definitions and (for masked models) the fitted policy.
You do not need the training dataset, optimizer checkpoints or a GPU on your PC.

Copy the folder from ARC, substituting your username and storage path:

```powershell
scp -r YOUR_UCALGARY_USERNAME@arc.ucalgary.ca:~/forecast-research/runs/experiment_v1/gui_models .
```

1. Click **Load model...** and choose an exported `.pt` file from `gui_models`.
   Select a model using the validation selection and evaluation reports. The
   filename identifies direct/MOS mode, architecture and seed; the GUI uses one
   selected seed at a time, not a seed ensemble.
2. Choose a **Location**.
3. Click **Load latest forecast inputs**. Initial downloads can take several
   minutes. The interface stays responsive and prevents conflicting edits while
   a request runs. A completed cycle is reused from the local cache.
4. Inspect the numerical inputs and satellite image, select forecast leads,
   and click **Calculate forecast**.
5. **Save results...** exports predictions, station ID, cycle/valid times, raw
   HRRR values, MOS corrections and optional acquisition cost to CSV.

Use this version of the project on ARC and locally. Model compatibility checks
reject a different package-source version. `best.pt` and `last.pt` are training
checkpoints, not portable GUI bundles. To export manually on ARC:

```bash
source arc/config.sh
"$FORECAST_ENV/bin/python" -m forecastprep.cli export-gui "$FORECAST_DATA" --runs "$FORECAST_RUNS" --output "$FORECAST_RUNS/gui_models"
```

## Locations and configurations

The default dropdown contains the 12 stations from `configs/stations_research.csv`.
**Load stations...** imports another station CSV with these exact columns:

```csv
station_id,latitude,longitude,iem_network
DEN,39.8328,-104.6575,CO_ASOS
COS,38.8058,-104.7008,CO_ASOS
```

Use the actual IEM station identifier, coordinates and network. This is a
station-based interface, not a worldwide city geocoder. HRRR coverage, nearby
grid cells, station observations and usable GOES imagery are required. Selecting
a new station does not establish model accuracy there. The GUI flags stations
outside the fitting station set, including the designated geographic holdouts.

**Load config...** accepts the project's YAML format, including its station CSV
reference. Relative CSV paths are resolved against the YAML file's directory.
The default `configs/live.yaml` retains the research features, geometry and
6/12/24/48-hour leads, and uses GOES-19 for contemporary acquisition. Dates in this
file are replaced by the selected recent cycle; historical train/val splits are
not applied to GUI downloads. A loaded model rejects incompatible feature or
image configurations. Loading a configuration clears previous inputs and results.

**Open saved inputs...** reopens a completed single-cycle folder, for example a
subfolder of `data/gui_cache`. Keep the whole folder, including its image files
and manifest. A full multi-cycle training archive is not a GUI input snapshot.
Switching stations clears manual edits and predictions; a snapshot from another
station cannot silently be used for the newly selected location.

## Time alignment and interpretation

"Latest" means the most recent configured 00/12 UTC initialization whose assumed
90-minute HRRR publication delay has elapsed. Observations and satellite scans
are restricted to their initialization-time cutoffs, matching training. It does
not combine a forecast from an old cycle with observations collected at click
time. All displayed times are UTC. Lead hours are measured from initialization,
so a short lead can already be in the past when you open the GUI; check the valid
time. The application does not automatically switch to hourly cycles that were
absent from training.

The 2021–24 research archive uses GOES-16. GOES-19 became operational GOES-East
on 7 April 2025 ([NOAA notice](https://www.noaa.gov/news-release/noaas-goes-19-satellite-now-operational-providing-critical-new-data-to-forecasters)).
The live configuration therefore uses GOES-19. The importer preserves the same
channels, physical units and geometry, but transfer accuracy across satellites
and years still needs evaluation. The GUI is a research interface, not a
validated operational weather service.

Double-click a numerical row to edit it. Edits apply to that lead's input vector
only, are held in memory, clear old predictions, and mark the saved prediction
as edited. Reopening or downloading inputs resets them. Editing a temperature
does not update the satellite scene; use edits as hypothetical scenarios.

MOS predicts a residual and adds it to HRRR exactly once. Direct models predict
temperature directly. The **Policy budget** and strategy affect masked models;
fixed models use their configured feature set. Costs are the experiment's
relative compute proxies, not seconds or a download-byte limit. Downloading a
snapshot prepares candidate inputs before policy inference.

## Failure handling

Missing required inputs, incompatible models/configurations, unsupported leads,
invalid edits and nonfinite predictions produce visible errors rather than a
fabricated forecast. Input downloads retain the pipeline's completed-cycle
checkpoints and cache; an interrupted unfinished cycle can be retried. Training
checkpoints and ARC recovery remain unchanged; see `docs/RECOVERY.md`.

Network access is needed only to fetch inputs. Model inference and inspection of
complete cached inputs work offline. The downloaded single-cycle folder is much
smaller than the research archive, although raw HRRR/GOES files can still be large.
You may remove obsolete `data/gui_cache` folders while the GUI is closed; the GUI
does not automatically erase them.
