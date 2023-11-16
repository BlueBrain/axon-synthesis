"""Store the general configuration values."""
import luigi


class Config(luigi.task.Config):
    """Config task to store global parameters."""

    # output_dir = luigi.parameter.OptionalPathParameter(
    #     description="The directory in which all the results will be exported",
    #     default=None,
    # )
    # input_data_type = luigi.ChoiceParameter(
    #     description=("The type of input data to use."),
    #     choices=["biological_morphologies", "white_matter"],
    #     default="biological_morphologies",
    # )
    # # atlas_path = luigi.parameter.OptionalPathParameter(
    # #     description="Atlas path", exists=True, default=None
    # # )
    # # atlas_region_filename = luigi.Parameter(
    # #     description="Atlas regions file.",
    # #     default="brain_regions",
    # # )
    # # atlas_flatmap_filename = luigi.OptionalParameter(
    # #     description="Atlas flatmap file.",
    # #     default=None,
    # # )
    # # atlas_hierarchy_filename = luigi.Parameter(
    # #     description="Atlas hierarchy file.",
    # #     default="hierarchy.json",
    # # )
    # white_matter_file = luigi.parameter.OptionalPathParameter(
    #     description="White matter file", default=None,
    # )

    # def __setattr__(self, name, value):
    #     if name == "output_dir":
    #         TaggedOutputLocalTarget.set_default_prefix(value)
    #     super().__setattr__(name, value)
