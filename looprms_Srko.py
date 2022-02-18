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


empty_directory('results/Srko')

# pathpkg = os.getcwd()
# dir1 = "build/FastGlobalRegistration"
# dir2 = "dataset"
#
# pathexec = os.path.join(pathpkg, dir1)
# pathdata = os.path.join(pathpkg, dir2)

for i in range(25):
    for j in range(10):
        proc = subprocess.run(["build/FastGlobalRegistration/FastGlobalRegistration",
                               "dataset/pairwise_no_noise_" +
                               str(i+1) + "_rot_05/features_0000.bin",
                               "dataset/pairwise_no_noise_" +
                               str(i+1) + "_rot_05/features_0001.bin",
                               "dataset/pairwise_no_noise_" + str(i+1) + "_rot_05/output_" + str(j+1) + ".txt"])

        proc2 = subprocess.run(["build/FastGlobalRegistration/Evaluation",
                                "dataset/pairwise_no_noise_" +
                                str(i+1) + "_rot_05/features_0000.bin",
                                "dataset/pairwise_no_noise_" +
                                str(i+1) + "_rot_05/features_0001.bin",
                                "dataset/pairwise_no_noise_" +
                                str(i+1) + "_rot_05/gt.log",
                                "dataset/pairwise_no_noise_" +
                                str(i+1) + "_rot_05/output_" + str(j+1) + ".txt",
                                "results/FastReg/no_noise_eval_collection_" + str(i+1) + ".txt"])
        # proc.wait()
        # proc.kill()
