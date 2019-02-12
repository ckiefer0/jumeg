from os import makedirs
import os.path as op
import mne
import numpy as np
from copy import copy
import glob
from jumeg.jumeg_volmorpher import auto_match_labels, volume_morph_stc


###############################################################################
# Subjects and conditions to do
###############################################################################

subjects_dir = '/Volumes/KieferFreeViewingData/FreeViewing/Data/'
pre_proc = 'bcc,tr,nr,fibp1-45,ar,'

condition_list = ['SEsac']

subject_list = ['211622']

###############################################################################
# Parameter for Morphing
###############################################################################
method = 'lcmv'  # 'MFT', 'uni' for MFT with uniform prob. dist., 'MNE', 'dSPM', 'lcmv'
stcs_dir = 'stcs_%s_ave' % method
template = 'fsaverage'
template_spacing = 5.0  # positions used for volume source space reconstruction.
subj_spacing = 5.0  # grid spacing used for volume source space reconstruction.
snr = 3.
mindist = 5.0
n_jobs = 1
interpolation_method = 'linear'

# baseline parameters
btmin = -0.2
btmax = 0.0

###############################################################################
# some morphing helper functions
###############################################################################


def set_directory(path=None):
    """
    check whether the directory exits, if no, create the directory
    ----------
    path : the target directory.

    """
    exists = op.exists(path)
    if not exists:
        makedirs(path)


def reduce_volume_labels(subj, indiv_spacing, subjects_dir, template, template_spacing,
                         volume_labels):
    """
    Check if for both the subject and the template vertices exist for
    each label. If a label is either missing vertices for the subject
    or the template, it is removed from volume_labels.

    Parameters:
    -----------
    subj : str
        ID of the subject.
    indiv_spacing :
        Grid space for the subject volume source space.
    subjects_dir : string
        The total path of all the subjects.
    template : string
        The subject name as the common brain.
    template_spacing : float
        Grid space for the template volume source space.
    volume_labels : list
        List with names of the volume labels.

    Returns:
    --------
    volume_labels_ : list
        List with names of the volume labels that have at least
        one vertice for both the subject and the template.
    """
    # copy volume_labels to avoid changing the input object
    volume_labels_ = copy(volume_labels)

    template_brain_dir = op.join(subjects_dir, template)

    # get list of labels and the lists of vertices for the subject brain
    fname_label_dict_subject = subj + '_vol-%.2f-src_vertno_labelwise.npy' % indiv_spacing
    fpath_label_dict_subject = op.join(subjects_dir, subj, fname_label_dict_subject)
    label_dict_subject = np.load(fpath_label_dict_subject).item()

    # get list of labels and the lists of vertices for the template brain
    fname_label_dict_template = template + '_vol-%.2f-src_vertno_labelwise.npy' % template_spacing
    fpath_label_dict_template = op.join(template_brain_dir, fname_label_dict_template)
    label_dict_template = np.load(fpath_label_dict_template).item()

    idx_to_pop = []
    for idx, label in enumerate(volume_labels_):
        pop = False
        if label_dict_subject[label].shape[0] < 1:
            print 'subj', label
            pop = True

        if label not in label_dict_template.keys():
            print 'temp', label, 'not found'
            pop = True
        elif label_dict_template[label].shape[0] < 1:
            print 'temp', label
            pop = True

        if pop:
            idx_to_pop.append(idx)

    # pop from back to front to avoid changing indices
    for idx in reversed(idx_to_pop):
        print 'popped', volume_labels_.pop(idx)

    return volume_labels_


def match_volume_labels(subject_list, subjects_dir, template='fsaverage', template_spacing=5.00):

    """
    Create the transformation matrices for volume source space morphing and
    save them to disc.

    Parameters:
    -----------
    subject_list : list
        List with IDs of all subjects.
    subjects_dir: string
        The total path of all the subjects.
    template: string
        The subject name as the common brain.
    template_spacing : float
        Grid space for the template volume source space.

    Returns:
    --------
    None
    """

    e_func = 'balltree'
    template_brain_dir = op.join(subjects_dir, template)

    for subj in subject_list:

        # indiv_spacing = make_indiv_spacing(subj, template, template_spacing, subjects_dir)
        indiv_spacing = 5.00
        fname_vsrc_subj = op.join(subjects_dir, subj, subj + '_vol-%.2f-src.fif' % indiv_spacing)
        fname_vsrc_template = op.join(subjects_dir, template, template + '_vol-%.2f-src.fif' % template_spacing)
        fname_trans = op.join(subjects_dir, subj, subj + '_' + template + '_vol-%.2f_lw-trans.npy' % template_spacing)

        # skip subject if file exists
        if op.isfile(fname_trans):
            continue

        volume_labels = mne.get_volume_labels_from_aseg(op.join(subjects_dir, subj, 'mri', 'aseg.mgz'))

        # get list of labels and the lists of vertices for the subject brain
        fname_label_dict_subject = subj + '_vol-%.2f-src_vertno_labelwise.npy' % indiv_spacing
        fpath_label_dict_subject = op.join(subjects_dir, subj, fname_label_dict_subject)
        label_dict_subject = np.load(fpath_label_dict_subject).item()

        # get list of labels and the lists of vertices for the template brain
        fname_label_dict_template = template + '_vol-%.2f-src_vertno_labelwise.npy' % template_spacing
        fpath_label_dict_template = op.join(template_brain_dir, fname_label_dict_template)
        label_dict_template = np.load(fpath_label_dict_template).item()

        # only consider labels with vertices in both subject and template brain
        volume_labels = reduce_volume_labels(subj, indiv_spacing, subjects_dir, template,
                                             template_spacing, volume_labels)

        # create transformation matrix for volume source space morphing
        auto_match_labels(fname_subj_src=fname_vsrc_subj, label_dict_subject=label_dict_subject,
                          fname_temp_src=fname_vsrc_template, label_dict_template=label_dict_template,
                          subjects_dir=subjects_dir, volume_labels=volume_labels, template_spacing=template_spacing,
                          e_func=e_func, fname_save=fname_trans, save_trans=True)


def morph_stc_volume(subject_list, pre_proc, method, subjects_dir, condition_list,
                     subj_spacing=5.00, template='fsaverage', template_spacing=5.00,
                     baseline_tmin=-0.2, baseline_tmax=0., interpolation_method=None):
    """
    Morph individual evoked volume source space STCs into the common brain
    space given by template.

    Parameter
    ------------------------------------
    subject_list : list of str
        List of subject IDs.
    pre_proc : str
        String specifying the performed preprocessing steps.
    method : str
        Method used to calculate the inverse solution.
    subjects_dir : string
        The total path of all the subjects.
    condition_list : list of str
        List with the conditions of interest.
    subj_spacing : float
        Grid space for the subject volume source space.
    template : string
        The subject name as the common brain.
    template_spacing : float
        Grid space for the template volume source space.
    baseline_tmin, baseline_tmax: float
        If 'baseline' is True, baseline is croped using this period.
    interpolation_method : str | None
        Only 'linear' seeems to be working for 3D data. 'balltree' and
        'euclidean' only work for 2D?.
    """
    normalize = True
    if interpolation_method is None:
        interpolation_method = 'linear'

    # on morphing to fsaverage in volume space
    # https://github.com/mne-tools/mne-python/issues/4819

    fn_list = []
    if condition_list is None:
        for subj in subject_list:
            dirname = op.join(subjects_dir, subj, 'stcs_%s_ave' % method)
            pattern = '*' + pre_proc + '*_evt_bc,ave,vol-%.2f-vl.stc' % subj_spacing
            fn_list.extend(glob.glob(op.join(dirname, pattern)))
    else:
        for subj in subject_list:
            dirname = op.join(subjects_dir, subj, 'stcs_%s_ave' % method)
            for cond in condition_list:
                pattern = '*' + pre_proc + cond + '_evt_bc,ave,vol-%.2f-vl.stc' % subj_spacing
                fn_list.extend(glob.glob(op.join(dirname, pattern)))

    for fpath in fn_list:
        # create file names
        print fpath

        stc_fname = op.basename(fpath)  # returns filename without folder
        subj = stc_fname.split('_')[0]
        run = int(stc_fname.split('_')[4])
        cond = stc_fname.split('_')[6].split(',')[-1]
        indiv_spacing = float(fpath.split('-')[2])
        stc_name = stc_fname.rsplit('-vl.stc')[0]
        template_brain_dir = op.join(subjects_dir, template)

        stc_path = op.join(template_brain_dir, method + '_STCs_ave', subj)

        set_directory(stc_path)

        volume_labels = mne.get_volume_labels_from_aseg(op.join(subjects_dir, subj, 'mri', 'aseg.mgz'))

        # only consider labels with vertices in both subject and template brain
        volume_labels = reduce_volume_labels(subj, indiv_spacing, subjects_dir, template,
                                             template_spacing, volume_labels)

        fname_vsrc_subj = op.join(subjects_dir, subj, subj + '_vol-%.2f-src.fif' % indiv_spacing)
        fname_vsrc_template = op.join(subjects_dir, template, template + '_vol-%.2f-src.fif' % template_spacing)

        fname_save_stc = op.join(stc_path, stc_name) + '-vl.stc'  # '-vl.stc' is necessary here
        fname_save_stc_base = op.join(stc_path, stc_name + '_baseline')  # '-vl.stc' not necessary here
        # perform the morphing
        stc_volmorphed = volume_morph_stc(fpath, subj, fname_vsrc_subj,
                                          volume_labels, template, fname_vsrc_template,
                                          cond, interpolation_method, normalize, subjects_dir,
                                          run=run, n_iter=None, unwanted_to_zero=True,
                                          label_trans_dic=None, fname_save_stc=fname_save_stc,
                                          save_stc=True, plot=True)

        stc_baseline = stc_volmorphed.crop(baseline_tmin, baseline_tmax)
        stc_baseline.save(fname_save_stc_base, ftype='stc')


print '>>> Match volume labels ....'
match_volume_labels(subject_list, subjects_dir, template=template,
                    template_spacing=template_spacing)
print '>>> FINISHED with label matching.'

morph_stc_volume(subject_list, pre_proc, method, subjects_dir, condition_list, template=template,
                 subj_spacing=subj_spacing, template_spacing=template_spacing, baseline_tmin=btmin,
                 baseline_tmax=btmax, interpolation_method=interpolation_method)
print '>>> FINISHED with morphed STC generation.'
