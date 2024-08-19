"""
Inference script
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
import rasterio as rio  # type: ignore[import-untyped]
from affine import Affine  # type: ignore[import-untyped]
from sensorsio.sentinel2 import Sentinel2
from sensorsio.sentinel2_l1c import Sentinel2L1C
from sensorsio.utils import bb_snap
from tqdm import tqdm


@dataclass(frozen=True)
class Chunk:
    source_area: rio.coords.BoundingBox
    target_area: rio.coords.BoundingBox


def generate_chunks(
    roi: rio.coords.BoundingBox,
    tile_size_in_meters: float,
    margin_in_meters: float,
) -> list[Chunk]:
    """
    Class initializer
    """
    # Find number of chunks in each dimension

    nb_chunks_x = np.ceil((roi.right - roi.left) / tile_size_in_meters)
    nb_chunks_y = np.ceil((roi.top - roi.bottom) / tile_size_in_meters)

    # Compute upper left corners of chunks
    chunks_x = roi.left + tile_size_in_meters * np.arange(0, nb_chunks_x)
    chunks_y = roi.bottom + tile_size_in_meters * np.arange(0, nb_chunks_y)
    # Generate the 2d grid of chunks upper left center
    chunks_x, chunks_y = np.meshgrid(chunks_x, chunks_y)

    # Flatten both list
    chunks_x = chunks_x.ravel()
    chunks_y = chunks_y.ravel()

    # Generate output chunk list
    chunks: list[Chunk] = []

    for cx, cy in zip(chunks_x, chunks_y):
        # Target area should not exceed roi
        target_area = rio.coords.BoundingBox(
            cx,
            cy,
            min(cx + tile_size_in_meters, roi.right),
            min(cy + tile_size_in_meters, roi.top),
        )
        # Source area is target area padded with margin
        source_area = rio.coords.BoundingBox(
            left=target_area.left - margin_in_meters,
            right=target_area.right + margin_in_meters,
            bottom=target_area.bottom - margin_in_meters,
            top=target_area.top + margin_in_meters,
        )
        chunks.append(Chunk(source_area, target_area))

    return chunks


from sentinel2_superresolution import __version__

__author__ = "Julien Michel"
__copyright__ = "CESBIO/CNES"
__license__ = "Apache_2.0"

_logger = logging.getLogger(__name__)


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="5m super-resolultion of Sentinel2 L2A products (Theia format)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"sentinel2_superresolution {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the input Sentinel2 L2A product directory or zip",
        required=True,
    )

    parser.add_argument(
        "--l1c", action="store_true", help="Input product is Sentinel2 L1C"
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        required=True,
        help="Output directory where to store output images",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to the onnx export of super-resolution model parameters",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models/carn_3x3x64g4sw_bootstrap.onnx",
        ),
    )
    parser.add_argument(
        "-ts",
        "--tile_size",
        type=int,
        default=1000,
        help="Tile size used for inference (expressed output reference system and resolution)",
    )

    parser.add_argument(
        "-ov",
        "--overlap",
        type=int,
        default=32,
        help="Overlap between tiles to use during inference",
    )

    parser.add_argument(
        "-roi",
        "--region_of_interest",
        type=float,
        nargs=4,
        default=None,
        help="Restrict region of interest to process (expressed in utm coordinates [left bottom right top])",
    )

    parser.add_argument(
        "-roip",
        "--region_of_interest_pixel",
        type=float,
        nargs=4,
        default=None,
        help="Restrict region of interest to process (expressed in in row/col [left bottom right tol])",
    )

    parser.add_argument(
        "--number_of_threads",
        type=int,
        default=8,
        help="Number of threads used for model inference",
    )

    parser.add_argument(
        "--bicubic",
        action="store_true",
        default=False,
        help="Also generate bicubic upsampled image",
    )

    parser.add_argument(
        "--gpu", action="store_true", help="Run inference on GPUs if available"
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    if args.l1c:
        s2_ds = Sentinel2L1C(args.input)
        # Bands that will be processed
        bands = [
            Sentinel2L1C.B2,
            Sentinel2L1C.B3,
            Sentinel2L1C.B4,
            Sentinel2L1C.B8,
            Sentinel2L1C.B5,
            Sentinel2L1C.B6,
            Sentinel2L1C.B7,
            Sentinel2L1C.B8A,
            Sentinel2L1C.B11,
            Sentinel2L1C.B12,
        ]
        level = "_L1C_"
    else:
        s2_ds = Sentinel2(args.input)
        # Bands that will be processed
        bands = [
            Sentinel2.B2,
            Sentinel2.B3,
            Sentinel2.B4,
            Sentinel2.B8,
            Sentinel2.B5,
            Sentinel2.B6,
            Sentinel2.B7,
            Sentinel2.B8A,
            Sentinel2.B11,
            Sentinel2.B12,
        ]
        level = "_L2A_"

    _logger.info(f"Will process {s2_ds}")
    _logger.info(f"Bounds: {s2_ds.bounds}, {s2_ds.crs}")
    _logger.info(f"Will use model {args.model}")

    # ONNX inference session options
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.number_of_threads
    so.inter_op_num_threads = args.number_of_threads
    so.use_deterministic_compute = True

    # Execute on cpu only
    ep_list = ["CPUExecutionProvider"]

    if args.gpu:
        _logger.info(f"Will run on GPU if available")
        ep_list.insert(0, "CUDAExecutionProvider")

    # Create inference session
    ort_session = ort.InferenceSession(args.model, sess_options=so, providers=ep_list)

    ro = ort.RunOptions()
    ro.add_run_config_entry("log_severity_level", "3")

    tile_size_in_meters = 5 * args.tile_size
    margin_in_meters = 5 * args.overlap

    # Read roi
    roi = s2_ds.bounds
    if args.region_of_interest_pixel is not None:
        logging.info(f"Pixel ROI set, will use it to define target ROI")
        roi_pixel = rio.coords.BoundingBox(*args.region_of_interest_pixel)
        roi = rio.coords.BoundingBox(
            left=s2_ds.bounds.left + 10 * roi_pixel.left,
            bottom=s2_ds.bounds.bottom + 10 * roi_pixel.bottom,
            right=s2_ds.bounds.left + 10 * roi_pixel.right,
            top=s2_ds.bounds.bottom + 10 * roi_pixel.top,
        )
    elif args.region_of_interest is not None:
        logging.info(
            f"ROI set, will use it to define target ROI. Note that provided ROI will be snapped to the 10m Sentinel-2 sampling grid."
        )
        roi = bb_snap(rio.coords.BoundingBox(*args.region_of_interest), align=10)

    # Adjust roi according to margin_in_meters
    roi = rio.coords.BoundingBox(
        left=max(roi.left, s2_ds.bounds.left + margin_in_meters),
        bottom=max(roi.bottom, s2_ds.bounds.bottom + margin_in_meters),
        right=min(roi.right, s2_ds.bounds.right - margin_in_meters),
        top=min(roi.top, s2_ds.bounds.top - margin_in_meters),
    )

    _logger.info(f"Will generate following roi : {roi}")
    chunks = generate_chunks(roi, tile_size_in_meters, margin_in_meters)
    _logger.info(f"Will process {len(chunks)} image chunks")

    # Output tiff profile
    geotransform = (roi[0], 5.0, 0.0, roi[3], 0.0, -5.0)
    transform = Affine.from_gdal(*geotransform)

    profile = {
        "driver": "GTiff",
        "height": int((roi[3] - roi[1]) / 5.0),
        "width": int((roi[2] - roi[0]) / 5.0),
        "count": 10,
        "dtype": np.int16,
        "crs": s2_ds.crs,
        "transform": transform,
        "nodata": -10000,
        "tiled": True,
    }

    # Ensure that ouptut directory exists
    os.makedirs(args.output_directory, exist_ok=True)

    # Derive output file name
    out_sr_file = os.path.join(
        args.output_directory,
        str(s2_ds.satellite.value)
        + "_"
        + s2_ds.date.strftime("%Y%m%d")
        + level
        + "T"
        + s2_ds.tile
        + "_5m_sisr.tif",
    )
    _logger.info(f"Super-resolved output image: {out_sr_file}")

    with rio.open(out_sr_file, "w", **profile) as rio_ds:
        for chunk in tqdm(
            chunks, total=len(chunks), desc="Super-resolution in progress ..."
        ):
            data_array = s2_ds.read_as_numpy(
                bounds=chunk.source_area,
                bands=bands,
                masks=None,
                resolution=10,
                scale=1.0,
                no_data_value=np.nan,
                algorithm=rio.enums.Resampling.cubic,
            )[0]

            output = ort_session.run(
                None, {"input": data_array[None, ...]}, run_options=ro
            )
            output = output[0][0, ...]

            output[np.isnan(output)] = -10000

            # Crop margin out
            if args.overlap != 0:
                cropped_output = output[
                    :, args.overlap : -args.overlap, args.overlap : -args.overlap
                ]
            else:
                cropped_output = output

            # Find location to write in ouptut image

            window = rio.windows.Window(
                int(np.floor((chunk.target_area.left - roi.left) / 5.0)),
                int(np.floor((roi.top - chunk.target_area.top) / 5.0)),
                int(np.ceil((chunk.target_area.right - chunk.target_area.left) / 5.0)),
                int(np.ceil((chunk.target_area.top - chunk.target_area.bottom) / 5.0)),
            )
            # Write ouptut image
            rio_ds.write(cropped_output, window=window)

    if args.bicubic:
        _logger.info("Will generate bicubic upsampled image as well")
        # Derive output file name
        out_bicubic_file = os.path.join(
            args.output_directory,
            str(s2_ds.satellite.value)
            + "_"
            + s2_ds.date.strftime("%Y%m%d")
            + "_L2A_"
            + "T"
            + s2_ds.tile
            + "_5m_bicubic.tif",
        )
        _logger.info(f"Bicubic output image: {out_bicubic_file}")
        chunks = generate_chunks(roi, tile_size_in_meters, margin_in_meters=0.0)
        with rio.open(out_bicubic_file, "w", **profile) as rio_ds:
            for chunk in tqdm(
                chunks, total=len(chunks), desc="Bicubic in progress ..."
            ):
                bicubic_array = s2_ds.read_as_numpy(
                    bounds=chunk.source_area,
                    bands=bands,
                    masks=None,
                    resolution=5,
                    scale=1.0,
                    no_data_value=np.nan,
                    algorithm=rio.enums.Resampling.cubic,
                )[0]

                bicubic_array[np.isnan(bicubic_array)] = -10000
                window = rio.windows.Window(
                    int(np.floor((chunk.target_area.left - roi.left) / 5.0)),
                    int(np.floor((roi.top - chunk.target_area.top) / 5.0)),
                    int(
                        np.ceil(
                            (chunk.target_area.right - chunk.target_area.left) / 5.0
                        )
                    ),
                    int(
                        np.ceil(
                            (chunk.target_area.top - chunk.target_area.bottom) / 5.0
                        )
                    ),
                )
                rio_ds.write(bicubic_array, window=window)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m sentinel2_superresolution.skeleton 42
    #
    run()
