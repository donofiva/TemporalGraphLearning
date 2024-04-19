from legacy.configuration import Configuration
from legacy.data import DataManager
from src.io.FileManager import FileManager
from src.io.Directory import Directory
from src.io.File import File


def main():

    # CONFIGURATION PHASE
    # Initialize configuration
    # TODO: This might become dynamic, now it's hardcoded
    configuration = Configuration(
        abnormal_values=[],
        field_to_filter_values={
            #Field.TURBINE: [1]
        }
    )

    # DATA LOADING PHASE
    # Resolve directories paths
    inputs_directory_path = FileManager.resolve_directory(Directory.INPUTS)

    # Resolve input paths
    turbines_file_path = FileManager.resolve_file(inputs_directory_path, File.TURBINES)
    timeseries_file_path = FileManager.resolve_file(inputs_directory_path, File.TIMESERIES)

    # Read input files
    data_manager = DataManager(configuration)
    turbines = data_manager.read_turbines_input_file(turbines_file_path)
    timeseries = data_manager.read_timeseries_input_file(timeseries_file_path)


    from legacy.abstraction.entities import Dataset
    dataset = Dataset(turbines, timeseries)

    # # Build turbines abstractions
    # turbines = AbstractionManager.build_turbines_abstraction(turbines_table)
    #
    # # Augment dataset with abstract entities and build temporal abstraction
    # timeseries_table = AbstractionManager.replace_keys_with_entities(timeseries_table)
    # timeseries = AbstractionManager.store_timeseries_abstraction(timeseries_table)
    # snapshots = AbstractionManager.build_snapshots_abstraction(timeseries_table)
    #
    # # MODELLING PHASE
    # # Data processing
    # from src.model.data_processing.AbnormalValuesManager import AbnormalValuesManager
    #
    # prova = AbnormalValuesManager(configuration.abnormal_values)
    # prova.evaluate_abnormal_values(snapshots)

    dataset.get_turbine_dataset().reset_index(drop=False).to_csv(
        FileManager.resolve_file(
            directory=FileManager.resolve_directory(Directory.OUTPUTS),
            file=File.TURBINES
        ),
        index=False
    )

    dataset.get_timeseries_dataset().reset_index(drop=False).to_csv(
        FileManager.resolve_file(
            directory=FileManager.resolve_directory(Directory.OUTPUTS),
            file=File.TIMESERIES
        ),
        index=False
    )


    # REPORTING PHASE
    # Store reports
    # ReportingManager.store_turbines_report(turbines)
    # ReportingManager.store_timeseries_report(turbines)


if __name__ == "__main__":
    main()