#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from cfnet.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "CFNetPlans"

    overwrite_plans = {
        'CFNetTrainerV2_2': ["CFNetPlans", "CFNetPlansisoPatchesInVoxels"], # r
        'CFNetTrainerV2': ["CFNetPlansnonCT", "CFNetPlansCT2", "CFNetPlansallConv3x3",
                            "CFNetPlansfixedisoPatchesInVoxels", "CFNetPlanstargetSpacingForAnisoAxis",
                            "CFNetPlanspoolBasedOnSpacing", "CFNetPlansfixedisoPatchesInmm", "CFNetPlansv2.1"],
        'CFNetTrainerV2_warmup': ["CFNetPlans", "CFNetPlansv2.1", "CFNetPlansv2.1_big", "CFNetPlansv2.1_verybig"],
        'CFNetTrainerV2_cycleAtEnd': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_cycleAtEnd2': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_reduceMomentumDuringTraining': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_graduallyTransitionFromCEToDice': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_independentScalePerAxis': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_Mish': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_Ranger_lr3en4': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_fp32': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_GN': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_momentum098': ["CFNetPlans", "CFNetPlansv2.1"],
        'CFNetTrainerV2_momentum09': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_DP': ["CFNetPlansv2.1_verybig"],
        'CFNetTrainerV2_DDP': ["CFNetPlansv2.1_verybig"],
        'CFNetTrainerV2_FRN': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_resample33': ["CFNetPlansv2.3"],
        'CFNetTrainerV2_O2': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_ResencUNet': ["CFNetPlans_FabiansResUNet_v2.1"],
        'CFNetTrainerV2_DA2': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_allConv3x3': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_ForceBD': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_ForceSD': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_LReLU_slope_2en1': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_lReLU_convReLUIN': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_ReLU': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_ReLU_biasInSegOutput': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_ReLU_convReLUIN': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_lReLU_biasInSegOutput': ["CFNetPlansv2.1"],
        #'CFNetTrainerV2_Loss_MCC': ["CFNetPlansv2.1"],
        #'CFNetTrainerV2_Loss_MCCnoBG': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_Loss_DicewithBG': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_Loss_Dice_LR1en3': ["CFNetPlansv2.1"],
        'CFNetTrainerV2_Loss_Dice': ["CFNetPlans", "CFNetPlansv2.1"],
        'CFNetTrainerV2_Loss_DicewithBG_LR1en3': ["CFNetPlansv2.1"],
        # 'CFNetTrainerV2_fp32': ["CFNetPlansv2.1"],
        # 'CFNetTrainerV2_fp32': ["CFNetPlansv2.1"],
        # 'CFNetTrainerV2_fp32': ["CFNetPlansv2.1"],
        # 'CFNetTrainerV2_fp32': ["CFNetPlansv2.1"],
        # 'CFNetTrainerV2_fp32': ["CFNetPlansv2.1"],

    }

    trainers = ['CFNetTrainer'] + ['CFNetTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'CFNetTrainerNewCandidate24_2',
        'CFNetTrainerNewCandidate24_3',
        'CFNetTrainerNewCandidate26_2',
        'CFNetTrainerNewCandidate27_2',
        'CFNetTrainerNewCandidate23_always3DDA',
        'CFNetTrainerNewCandidate23_corrInit',
        'CFNetTrainerNewCandidate23_noOversampling',
        'CFNetTrainerNewCandidate23_softDS',
        'CFNetTrainerNewCandidate23_softDS2',
        'CFNetTrainerNewCandidate23_softDS3',
        'CFNetTrainerNewCandidate23_softDS4',
        'CFNetTrainerNewCandidate23_2_fp16',
        'CFNetTrainerNewCandidate23_2',
        'CFNetTrainerVer2',
        'CFNetTrainerV2_2',
        'CFNetTrainerV2_3',
        'CFNetTrainerV2_3_CE_GDL',
        'CFNetTrainerV2_3_dcTopk10',
        'CFNetTrainerV2_3_dcTopk20',
        'CFNetTrainerV2_3_fp16',
        'CFNetTrainerV2_3_softDS4',
        'CFNetTrainerV2_3_softDS4_clean',
        'CFNetTrainerV2_3_softDS4_clean_improvedDA',
        'CFNetTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'CFNetTrainerV2_3_softDS4_radam',
        'CFNetTrainerV2_3_softDS4_radam_lowerLR',

        'CFNetTrainerV2_2_schedule',
        'CFNetTrainerV2_2_schedule2',
        'CFNetTrainerV2_2_clean',
        'CFNetTrainerV2_2_clean_improvedDA_newElDef',

        'CFNetTrainerV2_2_fixes', # running
        'CFNetTrainerV2_BN', # running
        'CFNetTrainerV2_noDeepSupervision', # running
        'CFNetTrainerV2_softDeepSupervision', # running
        'CFNetTrainerV2_noDataAugmentation', # running
        'CFNetTrainerV2_Loss_CE', # running
        'CFNetTrainerV2_Loss_CEGDL',
        'CFNetTrainerV2_Loss_Dice',
        'CFNetTrainerV2_Loss_DiceTopK10',
        'CFNetTrainerV2_Loss_TopK10',
        'CFNetTrainerV2_Adam', # running
        'CFNetTrainerV2_Adam_CFNetTrainerlr', # running
        'CFNetTrainerV2_SGD_ReduceOnPlateau', # running
        'CFNetTrainerV2_SGD_lr1en1', # running
        'CFNetTrainerV2_SGD_lr1en3', # running
        'CFNetTrainerV2_fixedNonlin', # running
        'CFNetTrainerV2_GeLU', # running
        'CFNetTrainerV2_3ConvPerStage',
        'CFNetTrainerV2_NoNormalization',
        'CFNetTrainerV2_Adam_ReduceOnPlateau',
        'CFNetTrainerV2_fp16',
        'CFNetTrainerV2', # see overwrite_plans
        'CFNetTrainerV2_noMirroring',
        'CFNetTrainerV2_momentum09',
        'CFNetTrainerV2_momentum095',
        'CFNetTrainerV2_momentum098',
        'CFNetTrainerV2_warmup',
        'CFNetTrainerV2_Loss_Dice_LR1en3',
        'CFNetTrainerV2_NoNormalization_lr1en3',
        'CFNetTrainerV2_Loss_Dice_squared',
        'CFNetTrainerV2_newElDef',
        'CFNetTrainerV2_fp32',
        'CFNetTrainerV2_cycleAtEnd',
        'CFNetTrainerV2_reduceMomentumDuringTraining',
        'CFNetTrainerV2_graduallyTransitionFromCEToDice',
        'CFNetTrainerV2_insaneDA',
        'CFNetTrainerV2_independentScalePerAxis',
        'CFNetTrainerV2_Mish',
        'CFNetTrainerV2_Ranger_lr3en4',
        'CFNetTrainerV2_cycleAtEnd2',
        'CFNetTrainerV2_GN',
        'CFNetTrainerV2_DP',
        'CFNetTrainerV2_FRN',
        'CFNetTrainerV2_resample33',
        'CFNetTrainerV2_O2',
        'CFNetTrainerV2_ResencUNet',
        'CFNetTrainerV2_DA2',
        'CFNetTrainerV2_allConv3x3',
        'CFNetTrainerV2_ForceBD',
        'CFNetTrainerV2_ForceSD',
        'CFNetTrainerV2_ReLU',
        'CFNetTrainerV2_LReLU_slope_2en1',
        'CFNetTrainerV2_lReLU_convReLUIN',
        'CFNetTrainerV2_ReLU_biasInSegOutput',
        'CFNetTrainerV2_ReLU_convReLUIN',
        'CFNetTrainerV2_lReLU_biasInSegOutput',
        'CFNetTrainerV2_Loss_DicewithBG_LR1en3',
        #'CFNetTrainerV2_Loss_MCCnoBG',
        'CFNetTrainerV2_Loss_DicewithBG',
        # 'CFNetTrainerV2_Loss_Dice_LR1en3',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
        # 'CFNetTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
