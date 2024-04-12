from tgl.configuration.Configuration import Configuration
from tgl.data.Field import Field
from tgl.data.DataManager import DataManager
from tgl.abstraction.AbstractionManager import AbstractionManager
from tgl.io.FileManager import FileManager
from tgl.io.Directory import Directory
from tgl.io.File import File
from tgl.reporting.ReportingManager import ReportingManager


def main():

    # Initialize configuration
    # TODO: This might become dynamic, now it's hardcoded
    configuration = Configuration(
        field_to_filter_values={Field.TURBINE: [1]}
    )

    # Resolve directories paths
    inputs_directory_path = FileManager.resolve_directory(Directory.INPUTS)

    # Resolve input paths
    locations_file_path = FileManager.resolve_file(inputs_directory_path, File.LOCATIONS)
    timeseries_file_path = FileManager.resolve_file(inputs_directory_path, File.TIMESERIES)

    # Read input files
    data_manager = DataManager(configuration)
    locations_table = data_manager.read_locations_input_file(locations_file_path)
    timeseries_table = data_manager.read_timeseries_input_file(timeseries_file_path)

    # Build abstractions
    turbines = AbstractionManager.build_turbines_abstraction(locations_table)
    _ = AbstractionManager.build_timeseries_abstraction(timeseries_table)
    grid_snapshots = AbstractionManager.build_grid_snapshots_abstraction(turbines)

    # Store reports
    ReportingManager.store_locations_report(turbines)
    ReportingManager.store_timeseries_report(turbines)


if __name__ == "__main__":
    main()