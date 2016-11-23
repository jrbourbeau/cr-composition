#!/usr/bin/env python

import os
import sys
import shutil
import argparse

# from composition.support_functions.checkdir import checkdir
import composition as comp


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description='Sets up "shebang" lines in scripts that use icecube software',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-t', '--toolset', dest='toolset',
                   default='py2-v1',
                   choices=['py2-v2', 'py2-v1', 'standard'],
                   help='Option to specify which cvmfs toolset to use')
    p.add_argument('-m', '--metaproject', dest='metaproject',
                   default='/path/to/metaproject/dir',
                   help='Path to metaproject containing ShowerLLH project')
    p.add_argument('--comp_data_dir', dest='comp_data_dir',
                   default='/path/to/comp/dir',
                   help='Path to composition storge directory')
    p.add_argument('--llh_dir', dest='llh_dir',
                   default='/path/to/llh/dir',
                   help='Path to ShowerLLH storge directory')
    args = p.parse_args()

    cwd = os.getcwd()
    filelist = ['ShowerLLH_scripts/make_tables/MakeHist.py',
                'ShowerLLH_scripts/make_tables/merger.py',
                'ShowerLLH_scripts/make_tables/normalize.py',
                'ShowerLLH_scripts/run_sim/MakeShowerLLH.py',
                'ShowerLLH_scripts/run_sim/MakeExtras.py',
                'ShowerLLH_scripts/run_sim/MakeMC.py',
                'ShowerLLH_scripts/run_sim/MakeLaputop.py',
                'ShowerLLH_scripts/run_sim/merge.py',
                'ShowerLLH_scripts/run_data/merge.py',
                'ShowerLLH_scripts/run_data/hists/save_hist.py',
                'ShowerLLH_scripts/run_data/MakeShowerLLH.py']

    # Set up 'shebang' line in appropreiate python scripts
    # to allow them to work as stand-alone IceTray scripts
    for file in filelist:
        with open(cwd + '/' + file, 'r') as original_file:
            current_toolset_line = original_file.readline()
            current_metaproject_line = original_file.readline()
            current_toolset = current_toolset_line.split('/')[5]
            current_metaproject = current_metaproject_line.split(' ')[1].replace('/build\n','')
            new_toolset_line = current_toolset_line.replace(current_toolset,
                                                            args.toolset)
            new_metaproject_line = current_metaproject_line.replace(
                current_metaproject,
                args.metaproject)
            with open(cwd + '/' + file, 'w') as new_file:
                new_file.write(new_toolset_line)
                new_file.write(new_metaproject_line)
                shutil.copyfileobj(original_file, new_file)

    # Set up support_functions/paths.py
    with open(cwd + '/support_functions/paths.py', 'r') as original_paths:
        lines = original_paths.readlines()
        current_comp_data_dir_line = lines[7]
        current_comp_data_dir = current_comp_data_dir_line.split()[-1].strip()
        new_comp_data_dir_line = current_comp_data_dir_line.replace(
            current_comp_data_dir, '"'+args.comp_data_dir+'"')
        current_llhdir_line = lines[8]
        current_llhdir = current_llhdir_line.split()[-1].strip()
        new_llhdir_line = current_llhdir_line.replace(current_llhdir,
            '"'+args.llh_dir+'"')
        current_metaproject_line = lines[6]
        current_metaproject = current_metaproject_line.split()[-1].strip()
        new_metaproject_line = current_metaproject_line.replace(
            current_metaproject,
            '"'+args.metaproject+'"')

        with open(cwd + '/support_functions/paths.py', 'w') as new_paths:
            lines[7] = new_comp_data_dir_line
            lines[8] = new_llhdir_line
            lines[6] = new_metaproject_line
            new_paths.writelines(lines)
