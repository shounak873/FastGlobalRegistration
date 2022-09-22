#!/usr/bin/env python

import subprocess
import os
import os.path
import glob
import shutil


def remove_thing(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def empty_directory(path):
    for i in glob.glob(os.path.join(path, '*')):
        remove_thing(i)


empty_directory('/home/navlab-shounak/Desktop/RegResults/FastReg/noisy')
empty_directory('/home/navlab-shounak/Desktop/RegResults/FastReg/noisy_output')

# pathpkg = os.getcwd()
# dir1 = "build/FastGlobalRegistration"
# dir2 = "dataset"
#
# pathexec = os.path.join(pathpkg, dir1)
# pathdata = os.path.join(pathpkg, dir2)

for i in range(25):
    proc = subprocess.run(["build/FastGlobalRegistration/FastGlobalRegistration",
                           "dataset/pairwise_noise_xyz_level_02_" +
                           str(i+1) + "_rot_05/features_0000.bin",
                           "dataset/pairwise_noise_xyz_level_02_" +
                           str(i+1) + "_rot_05/features_0001.bin",
                           "/home/navlab-shounak/Desktop/RegResults/FastReg/noisy_output/" + str(i+1) + "_rot_05_output" + ".txt"])

    proc2 = subprocess.run(["build/FastGlobalRegistration/Evaluation",
                            "dataset/pairwise_noise_xyz_level_02_" +
                            str(i+1) + "_rot_05/features_0000.bin",
                            "dataset/pairwise_noise_xyz_level_02_" +
                            str(i+1) + "_rot_05/features_0001.bin",
                            "dataset/pairwise_noise_xyz_level_02_" +
                            str(i+1) + "_rot_05/gt.log",
                            "/home/navlab-shounak/Desktop/RegResults/FastReg/noisy_output/" +
                            str(i+1) + "_rot_05_output" + ".txt",
                            "/home/navlab-shounak/Desktop/RegResults/FastReg/noisy/noisy_eval_collection_" + str(i+1) + ".txt"])
    # proc.wait()
    # proc.kill()
