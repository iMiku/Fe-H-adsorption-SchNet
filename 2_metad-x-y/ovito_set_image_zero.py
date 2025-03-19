from ovito.io import import_file, export_file
from ovito.modifiers import ComputePropertyModifier
import argparse

def set_image_zero(file_name):
    pipeline = import_file(file_name)
    pipeline.modifiers.append(ComputePropertyModifier(
        output_property = 'Periodic Image',
        expressions = ["0", "0", "0"]               )
                             )
    export_file(pipeline, "np_"+file_name, "lammps/data", atom_style="bond")

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A simple script that sets all image flags to zero.")

    # Add a positional argument
    parser.add_argument("data_file", type=str, help="name of the data file")

    # Parse the command-line arguments
    args = parser.parse_args()
    set_image_zero(args.data_file)

if __name__ == "__main__":
    main()
