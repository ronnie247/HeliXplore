# HeliXplore: A Python package for analyzing multi-strand helix deformations

For and post-submission to Journal of Open Source Software (`v1.0.0`).

HeliXplore is an open-source Python tool for quantitative analysis of deformations in multi-strand helices, using molecular dynamics trajectories.

## Requirements

`HeliXplore` requires Python 3.7+ and the following libraries:  `numpy`, `scipy`, `pandas`, `matplotlib`. These dependencies will be automatically checked for at the beginning of each run.

## Citation

If you use `HeliXplore` in your work, please cite the accompanying publication:

`Mondal, R., & Vaissier Welborn, V.* HeliXplore: A Python package for analyzing multi-strand helix deformations. Journal Name, Year. [DOI/link to be added when available]`

## Help Information

Here is the help information you should see after running `python HeliXplore.py --help`

```bash
usage: HeliXplore.py [-h] [-f FILE] [-s STRANDS] [-l LENGTH] [--atom-name [ATOM_NAME ...]] [--atom-type [ATOM_TYPE ...]] [--no-plot] [-w WORKDIR] [-v] [--check-file] [filepath]

Analyze molecular trajectories with various deformations

positional arguments:
  filepath              Path to trajectory file (can be absolute, relative, or just filename)

options:
  -h, --help            show this help message and exit
  -f, --file FILE       Path to trajectory file (alternative to positional argument)
  -s, --strands STRANDS
                        Number of strands
  -l, --length LENGTH   Strand length
  --atom-name [ATOM_NAME ...]
                        Atom name(s) to filter. For PDB files (columns 13-16), include spaces (e.g., " CA " " N " " C "). For ARC files, spaces are optional.
  --atom-type [ATOM_TYPE ...]
                        Atom type(s) to filter (e.g., 129 130 156). Only applies to ARC files. Sixth column in ARC file. Can specify multiple types. Takes priority over atom name.
  --no-plot             Disable default plotting
  -w, --workdir WORKDIR
                        Working directory (default: directory containing the trajectory file)
  -v, --verbose         Enable verbose output
  --check-file          Check if trajectory file exists before running analysis

    Examples:
    python HeliXplore.py /path/to/trajectory.arc --strands 5
    python HeliXplore.py data/trajectory.arc --strands 2 --length 15 --atom-type 50 51 52
    python HeliXplore.py /path/to/trajectory.pdb --strands 3 --length 20 --atom-name " CA " " N " "C" --atom-type 401 402
    
    *** FOR PDB FILES: Atom names must include proper spacing (4 characters).
    If not provided, HeliXplore.py will autocorrect. Note that autocorrect may not always be accurate.
    python HeliXplore.py trajectory.pdb --strands 2 --length 15 --atom-name " CA " " N  " " C  "

    *** NOTE: If the length of strand of the multi-helix size is not provided, HeliXplore.py will autocalculate. 
    Make sure your inputs are properly taken into account. Make sure to input the number of strands.
    False validations can occur if the inputs are wrong but the product matches the helix units.
    HeliXplore.py assumes all strands have the same number of units.

    *** RUNNING EXAMPLES: To replicate the examples provided on GitHub, run:
    python HeliXplore.py singlehelix.arc --strands 1 --atom-type 401 402
    python HeliXplore.py doublehelix.arc --strands 2 --atom-name "CA" "N"
    python HeliXplore.py triplehelix.arc --strands 3 --length 15
        
--------------------------------------------------

Output will be directed to: tempfile_YYYY-MM-DD_hh-mm-ss.out
--------------------------------------------------
```

where the tempfile is timestamped.

## Inputs to `HeliXplore`

The complete `HeliXplore` usage is as follows:

```bash
usage: HeliXplore.py [-h] [-f FILE] [-s STRANDS] [-l LENGTH] [--atom-name [ATOM_NAME ...]] [--atom-type [ATOM_TYPE ...]] [--no-plot] [-w WORKDIR] [-v] [--check-file] [filepath]
```

These inputs are mandatory: `--file` and `--strands`. `HeliXplore` will autocalculate the length of each strand by assuming all strands are of equal length.

Other useful inputs are:

`--length` - Strand length of each helix. If not provided, `HeliXplore` will autocalculate from the number of available helix units and `--strands`.

`--atom-name` - Atom name(s) of the atoms that form the backbone of a strand in the helix. There is no condition on the number of atom names that can be provided. This is the only way to select atoms if you use a PDB as the input file. With the PDB, include spaces to match columns 13-16. Example usage: `--atom-name " CA " " N " " C "`. If you use a TINKER `.arc` file, the leading and trailing spaces will be automatically removed. 

WARNING: `HeliXplore` auto-corrects for standard PDB atom names, however, this autocorrect might not be accurate. For example: `--atom-name " CA " " N" "C "` autocorrects to `--atom-name " CA " " N  " " C  "`. Check your PDB for the correct spacing in your atom name.

`--atom-type` - Atom type(s) of the atoms that form the backbone of a strand in the helix. There is no condition on the number of atom types that can be provided. This only applies to ARC files. When both `--atom-name` and `--atom-type` are provided, `--atom-type` takes priority. Example usage: `--atom-type 129 130 156`.

DEFAULT: If neither `--atom-name` nor `--atom-type` is provided, `HeliXplore` defaults to `--atom-name "CA"`.

`--no-plot` - To disable the default plotting.

`--workdir` - Working directory, the default is the directory containing the trajectory file.

## Example Usage

Example trajectories are provided to demonstrate how `HeliXplore` works.

To analyze the example single (amylose), double (collagen) and triple (collagen) helix trajectories:

```bash
python HeliXplore.py singlehelix.arc --strands 1 --atom-type 401 402
python HeliXplore.py doublehelix.arc --strands 2 --atom-name "CA" "N"
python HeliXplore.py triplehelix.arc --strands 3 --length 15
```

The expected outputs for the example commands are provided in the `benchmark` directory, which contains input `.arc` files and their corresponding output folders for single-, double-, and triple-helix cases.

```
benchmark/
├── helpmessage.out
├── singlehelix/
│   ├── singlehelix.arc
│   └── output/
│       ├── output_directory/
│       └── output_tempfile.out
├── doublehelix/
│   ├── doublehelix.arc
│   └── output/
│       ├── output_directory/
│       └── output_tempfile.out
└── triplehelix/
    ├── triplehelix.arc
    └── output/
        ├── output_directory/
        └── output_tempfile.out
```

## Output

`HeliXplore` will have a timestamped `.out` file for every time the command is run (including `--help` runs). This filename will be named `tempfile_YYYY-MM-DD_hh-mm-ss.out`. When run with a trajectory file, an output folder will also be created, named `HeliXplore_YYYY-MM-DD_hh-mm-ss`, where all the output files will be created.

For single and multi-helix systems, `HeliXplore` runs Section 1 and Section 2 of the code. The following are the output files:

```bash
Section1_multi_helix_deform_results_plot_over_time.png
Section1_multi_helix_deform_results_plot.png
Section1_multi_helix_deform_results_time_per_strand.dat
Section1_multi_helix_deform_results_time.dat
Section1_multi_helix_deform_results_unit.dat
Section2_multi_helix_deformation_table.dat
```

For triple-helix systems, `HeliXplore` also runs Section 3, with additional output files as:

```bash
Section3_triad_deformation_by_time.dat
Section3_triad_deformation_by_time.png
Section3_triad_deformation_by_unit.dat
Section3_triad_deformation_by_unit.png
Section3_triad_deformation.dat
```

## Community Guidelines

Users are free to copy, modify, and use `HeliXplore` for their own research. Contributions, improvements, and extensions are welcome. Please include brief comments in your code to describe any added functionality. Bug reports and feature requests can be submitted through the GitHub Issues page.

## License

HeliXplore is distributed under the BSD 3-Clause License. See the `LICENSE` file in this repository for the full text.


