"""Compute the long-range trunk properties after clustering."""
import json
import logging

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from neurom import COLS
from neurom import features
from neurom import load_morphology
from neurom.apps import morph_stats

from axon_synthesis.PCSF.clustering import ClusterTerminals
from axon_synthesis.utils import get_axons

logger = logging.getLogger(__name__)


class TrunkPropertiesOutputLocalTarget(TaggedOutputLocalTarget):
    """Target for clustering outputs."""

    __prefix = "trunk_properties"  # pylint: disable=unused-private-member


def angle_between_vectors(p1, p2):
    """Computes the angle in radians between vectors 'p1' and 'p2'.

    Normalizes the input vectors and computes the relative angle
    between them.

        >>> angle_between((1, 0), (0, 1))
        1.5707963267948966
        >>> angle_between((1, 0), (1, 0))
        0.0
        >>> angle_between((1, 0), (-1, 0))
        3.141592653589793
    """
    if p1.shape == p2.shape and np.equal(p1, p2).all():
        return 0.0
    p1 = np.array(p1, ndmin=2)[:, COLS.XYZ]
    p2 = np.array(p2, ndmin=2)[:, COLS.XYZ]
    if p1.shape[0] == 1:
        p1 = np.repeat(p1, p2.shape[0], axis=0)
    if p2.shape[0] == 1:
        p2 = np.repeat(p2, p1.shape[0], axis=0)
    v1 = p1 / np.linalg.norm(p1, axis=1)[:, np.newaxis]
    v2 = p2 / np.linalg.norm(p2, axis=1)[:, np.newaxis]
    dot = np.einsum("ij,ij->i", v1, v2)
    return np.arccos(np.clip(dot, -1.0, 1.0))


@features.feature(shape=(...,), namespace=features.NameSpace.NEURITE)
def segment_angles(neurite, reference=[0, 1, 0]):
    res = features.neurite._map_segments(section_segment_angles, neurite)
    return res


def vector(p1, p2):
    """Compute vector between two 3D points.

    Args:
        p1, p2: indexable objects with
        indices 0, 1, 2 corresponding to 3D cartesian coordinates.

    Returns:
        3-vector from p1 - p2
    """
    return np.subtract(p1[..., COLS.XYZ], p2[..., COLS.XYZ])


def section_segment_angles(section, reference=[0, 1, 0]):
    """Angles between the segments of a section and a reference vector."""
    ref = np.array(reference)
    seg_vectors = vector(section.points[1:], section.points[:-1])
    directions = angle_between_vectors(seg_vectors, ref)
    return directions


class LongRangeTrunkProperties(luigi_tools.task.WorkflowTask):
    """Task to cluster the terminals."""

    morphology_paths = luigi.OptionalPathParameter(
        description="Path to the CSV file containing the paths to the trunk morphologies.",
        default=None,
        exists=True,
    )

    def requires(self):
        return ClusterTerminals()

    def run(self):
        # Load morph paths
        logger.info("Extract trunk properties")
        morphology_paths = pd.read_csv(
            self.morphology_paths or self.input()["trunk_morphology_paths"].path,
            dtype={"morph_file": str},
        )
        long_range_trunk_props = []

        for idx, row in morphology_paths.iterrows():
            logger.info("Extract trunk properties from %s", row["morph_path"])
            trunk_morph = load_morphology(row["morph_path"])

            # Compute long-range trunk features that will be used for smoothing and jittering
            long_range_trunks = get_axons(trunk_morph)
            for num, axon in enumerate(long_range_trunks):
                trunk_stats = morph_stats.extract_stats(
                    axon,
                    {
                        "neurite": {
                            "segment_lengths": {"modes": ["raw", "mean", "std"]},
                            "segment_meander_angles": {"modes": ["raw", "mean", "std"]},
                            "segment_angles": {"modes": ["raw"]},
                            "segment_path_lengths": {"modes": ["raw"]},
                        }
                    },
                )["axon"]
                long_range_trunk_props.append(
                    (
                        row["morph_file"],
                        num,
                        json.dumps(np.array(trunk_stats["raw_segment_lengths"]).tolist()),
                        trunk_stats["mean_segment_lengths"],
                        trunk_stats["std_segment_lengths"],
                        json.dumps(np.array(trunk_stats["raw_segment_meander_angles"]).tolist()),
                        trunk_stats["mean_segment_meander_angles"],
                        trunk_stats["std_segment_meander_angles"],
                        json.dumps(np.array(trunk_stats["raw_segment_angles"]).tolist()),
                        json.dumps(np.array(trunk_stats["raw_segment_path_lengths"]).tolist()),
                    )
                )

        # Export long-range trunk properties
        trunk_props_df = pd.DataFrame(
            long_range_trunk_props,
            columns=[
                "morph_file",
                "axon_id",
                "raw_segment_lengths",
                "mean_segment_lengths",
                "std_segment_lengths",
                "raw_segment_meander_angles",
                "mean_segment_meander_angles",
                "std_segment_meander_angles",
                "raw_segment_angles",
                "raw_segment_path_lengths",
            ],
        )
        trunk_props_df.sort_values(["morph_file", "axon_id"], inplace=True)
        trunk_props_df.to_csv(self.output().path, index=False)
        logger.info("Exported trunk properties to %s", self.output().path)

    def output(self):
        return TrunkPropertiesOutputLocalTarget("trunk_properties.csv", create_parent=True)
