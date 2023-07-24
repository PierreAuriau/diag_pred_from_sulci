from datasets.open_bhb import OpenBHB
import os
import numpy as np
import logging

logger = logging.getLogger()

class BHB(OpenBHB):
    """
        BHB-10K Dataset written in a torchvision-like manner. It is memory-efficient, taking advantage of
        memory-mapping implemented with NumPy. It is an extension of OpenBHB. It comes with 2 differents schemes:
            - [5-fold CV scheme stratified on age+sex+site] Not available yet
            - Train/Validation/Test split
        ... 2 pre-processings:
            - Quasi-Raw
            - VBM
        ... And 2 differents tasks:
            - Age prediction (regression)
            - Sex prediction (classification)
        ... With meta-data:
            - unique identifier across pre-processing and split (participant_id, session, run, study)
            - TIV + ROI measures based on Neuromorphometrics atlas
    Attributes:
          * target, list[int]: labels to predict
          * target_mapping: dict(int: str): each label is associated with its original name (if any)
          * shape, tuple: shape of the data
          * metadata: pd DataFrame: TIV + ROI measures extracted for each image
          * id: pandas DataFrame, each row contains a unique identifier for an image

    """

    def _set_dataset_attributes(self):
        self._studies = ['abide1', 'abide2', 'hcp', 'ixi', 'npc', 'rbp', 'oasis3', 'gsp', 'icbm', 'localizer',
                         'mpi-leipzig', 'corr', 'nar', 'biobd', 'schizconnect-vip', 'bsnip']
        self._train_val_test_scheme = "train_val_test_test-intra_open_bhb-extended_stratified.pkl"
        self._cv_scheme = None
        self._mapping_sites = "mapping_site_name-class_extended.pkl"
        self._npy_files = {"vbm": "%s_t1mri_mwp1_gs-raw_data64.npy",
                           "quasi_raw": "%s_t1mri_quasi_raw_data32_1.5mm_skimage.npy",
                           "skeleton": "%s_t1mri_skeleton_data64.npy"}
        self._pd_files = {"vbm": "%s_t1mri_mwp1_participants.csv",
                          "quasi_raw": "%s_t1mri_quasi_raw_participants.csv",
                          "skeleton": "%s_t1mri_skeleton_participants.csv"}
        self._preproc_folders = {"vbm": "cat12vbm", "quasi_raw": "quasi_raw", "skeleton": "morphologist"}

    def _check_integrity(self):
        if self.scheme_name == "cv":
            raise NotImplementedError("No CV scheme implemented for BHB (yet).")
        is_complete = os.path.isdir(self.root)
        is_complete &= os.path.isfile(os.path.join(self.root, self._train_val_test_scheme))
        is_complete &= os.path.isfile(os.path.join(self.root, self._mapping_sites))
        dir_files = {folder: [self._npy_files[preproc], self._pd_files[preproc]]
                     for preproc, folder in self._preproc_folders.items()}
        for (dir, files) in dir_files.items():
            for file in files:
                for db in self._studies:
                    is_complete &= os.path.isfile(os.path.join(self.root, dir, file % db))
        return is_complete

    def _extract_metadata(self, df):
        """
        :param df: pandas DataFrame
        :return: TIV and tissue volumes defined by the Neuromorphometrics atlas
        """
        metadata = ["tiv"] + [k for k in df.keys() if "GM_Vol" in k or "WM_Vol" in k or "CSF_Vol" in k]
        assert len(metadata) == 288, "Missing meta-data values (%i != %i)"%(len(metadata), 288)
        assert set(metadata) <= set(df.keys()), "Missing meta-data columns: {}".format(set(metadata) - set(df.keys))
        if df[metadata].isna().sum().sum() != 0:
            print("Warning: NaN values found in meta-data", flush=True)
        return df[metadata]

    def _extract_mask(self, df, unique_keys):
        """
        :param df: pandas DataFrame
        :param unique_keys: list of str
        :return: a binary mask indicating, for each row, if the participant belongs to the current scheme or not.
        """
        # TODO: correct this hack in the final version
        df = df.copy()
        df.loc[df['run'].isna(), 'run'] = 1
        df.loc[df['session'].isna(), 'session'] = 1
        if df['run'].dtype == np.float:
            df['run'] = df['run'].astype(int)
        clinical_studies = ['BIOBD', 'BSNIP', 'SCHIZCONNECT-VIP', 'PRAGUE']
        df.loc[df['session'].eq('V1') & df['study'].isin(clinical_studies), 'session'] = 1
        df.loc[df['session'].eq('v1') & df['study'].isin(clinical_studies), 'session'] = 1

        _source_keys = df[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        _target_keys = self.scheme[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        mask = _source_keys.isin(_target_keys).values.astype(np.bool)
        return mask

    def __str__(self):
        if self.fold is not None:
            return "BHB-%s-%s-%s-%s"%(self.preproc, self.scheme_name, self.split, self.fold)
        return "BHB-%s-%s-%s"%(self.preproc, self.scheme_name, self.split)


class HCPDataset(OpenBHB):
    def _set_dataset_attributes(self):
        self._studies = ['hcp']
        self._train_val_test_scheme = "train_val_test_hcp_stratified.pkl"
        self._cv_scheme = None
        self._mapping_sites = "mapping_site_name-class_extended.pkl"
        self._npy_files = {"vbm": "%s_t1mri_mwp1_gs-raw_data64.npy",
                           "quasi_raw": "%s_t1mri_quasi_raw_data32_1.5mm_skimage.npy",
                           "skeleton": "%s_t1mri_skeleton_data32.npy"}
        self._pd_files = {"vbm": "%s_t1mri_mwp1_participants.csv",
                          "quasi_raw": "%s_t1mri_quasi_raw_participants.csv",
                          "skeleton": "%s_t1mri_skeleton_participants.csv"}
        self._preproc_folders = {"skeleton": "morphologist"}

    def _check_integrity(self):
        is_complete = os.path.isdir(self.root)
        is_complete &= os.path.isfile(os.path.join(self.root, self._train_val_test_scheme))
        is_complete &= os.path.isfile(os.path.join(self.root, self._mapping_sites))
        dir_files = {folder: [self._npy_files[preproc], self._pd_files[preproc]]
                     for preproc, folder in self._preproc_folders.items()}
        logger.debug(f"Dir files : {dir_files}")
        logger.debug(f"Preproc folder : {self._preproc_folders}")
        logger.debug(f"Studies : {self._studies}")
        for (dir, files) in dir_files.items():
            for file in files:
                for db in self._studies:
                    is_complete &= os.path.isfile(os.path.join(self.root, dir, file % db))
        return is_complete

    def _extract_metadata(self, df):
        """
        :param df: pandas DataFrame
        :return: TIV and tissue volumes defined by the Neuromorphometrics atlas
        """
        metadata = ["tiv"] + [k for k in df.keys() if "GM_Vol" in k or "WM_Vol" in k or "CSF_Vol" in k]
        if len(metadata) != 288:
            logger.warning("Missing meta-data values (%i != %i)"%(len(metadata), 288))
        if set(metadata) > set(df.keys()):
            logger.warning("Missing meta-data columns: {}".format(set(metadata) - set(df.keys)))
        if df[metadata].isna().sum().sum() != 0:
            logger.warning("NaN values found in meta-data")
        return df[metadata]
