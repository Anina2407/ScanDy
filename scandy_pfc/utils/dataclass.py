import os
import numpy as np
import pandas as pd
import imageio
import yaml
from skimage.transform import resize

from .functions import anisotropic_centerbias


class Dataset:
    """
    This class loads and handles the human scanpath data (gt_foveation_df), the
    videos of the dataset and all precomputed maps, including object segmentations,
    optical flow, and different saliency meassures.
    Attributes contain important information about the data, like the size of
    the videos (#frames, size in x & y) and the conversion from pixels to degrees
    visual angle (dva).

    Required information in the dataset:
        * PATH: Path to the data where all if stored in predefined schema (see below)
        * FPS: Frames per second of the videos
        * PX_TO_DVA: Conversion from pixels to degrees visual angle
        * gt_foveation_df: Dataframe with the ground truth scanpaths

    The following information & paths are derived or set if not explicitely given(-> indicates how it is inferred):
        * videos: -> {PATH}videos/, original videos (only used for visualization)
        * featuremaps: -> {PATH}featuremaps/, maps for features (have subfolders for different feature extraction methods, like 'molin')
        * objectmasks: -> {PATH}polished_segmentation/, Object segmentation masks (polished, px values are time consistent object ids)
        * flowmaps: -> {PATH}optical_flow/, Optical flow maps
        * nssmaps: -> {PATH}gt_fov_maps_333/, Ground truth fixation maps, normalized for calculating NSS
        * used_videos: -> names in objectmasks, list of videos used for modeling, usually limited by segmentations
        * FRAMES_ALL_VIDS: -> objectmasks[used_videos].shape[0], Number of frames either given for all (int) or inferred for each (list)
        * VID_SIZE_Y, VID_SIZE_X: -> objectmasks[0].shape[1:] are assumed to be the same for all videos (for convenience).
        * video_frames: -> FRAMES_ALL_VIDS, Number of frames for each video (dict)
        * RADIUS_OBJ_GAZE: Tolerance radius of the object around the gaze point (in dva), default: 1.0
        * trainset: List of videos used for training, default: used_videos
        * testset: List of videos used for testing, default: None
        * gt_fovframes_nss_df: Dataframe with NSS values of the ground truth scanpaths, only needed if NSS is of interest
    """

    def __init__(self, dataconfig):
        """
        Load the important information from the provided dataset.
        :param dataconfig: Dictionary that contains the most important info about the dataset.
        :type dataconfig: dict or str (path to yaml file)
        """
        # load the dataconfig, either as dictionary or from a yaml file
        if isinstance(dataconfig, dict):
            datadict = dataconfig
        elif isinstance(dataconfig, str):
            assert dataconfig.endswith(".yaml") or dataconfig.endswith(
                ".yml"
            ), "Path is not a YAML file!"
            datadict = self.load_yaml(dataconfig)
            assert isinstance(datadict, dict), "YAML file was not converted to dict!"
        else:
            raise TypeError(
                "Input must be a dictionary or a string path to a YAML file"
            )

        # Inputs that cannot easily be derived from maps must be given:
        assert "FPS" in datadict, f"FPS has to be provided as key in dataconfig!"
        self.FPS = datadict["FPS"]
        assert (
            "PX_TO_DVA" in datadict
        ), f"PX_TO_DVA has to be provided as key in dataconfig!"
        self.PX_TO_DVA = datadict["PX_TO_DVA"]
        self.DVA_TO_PX = 1.0 / self.PX_TO_DVA

        # Path to the data where all if stored in predefined schema (see below)
        assert (
            "PATH" in datadict
        ), f"PATH to data has to be provided as key in dataconfig!"
        self.PATH = datadict["PATH"]
        if "VID_EXTENSION" in datadict:
            self.vid_ext = datadict["VID_EXTENSION"]
        else:
            self.vid_ext = "mpg"
        # derive other paths if not explicitely given...
        # maps for features (have subfolders for different feature extraction methods, like 'molin')
        if "featuremaps" in datadict:
            self.featuremaps = datadict["featuremaps"]
        else:
            self.featuremaps = f"{self.PATH}featuremaps/"
        # maps for top-down guidance 
        if "topdownmaps" in datadict:
            self.topdownmaps = datadict["topdownmaps"]
        else:
            self.topdownmaps = f"{self.PATH}topdownmaps/"
        # Object segmentation masks (polished)
        if "objectmasks" in datadict:
            self.objectmasks = datadict["objectmasks"]
        else:
            self.objectmasks = f"{self.PATH}polished_segmentation/"
        # Optical flow
        if "flowmaps" in datadict:
            self.flowmaps = datadict["flowmaps"]
        else:
            self.flowmaps = f"{self.PATH}optical_flow/"
        # Ground truth fixation maps, normalized for calculating NSS
        if "nssmaps" in datadict:
            self.nssmaps = datadict["nssmaps"]
        else:
            self.nssmaps = f"{self.PATH}gt_fov_maps_333/"
        # Original videos in RGB, just used for visualization
        if "videoframes" in datadict:
            self.videoframes = datadict["videoframes"]
        else:
            self.videoframes = f"{self.PATH}videos/"
        # list of videos used for modeling, usually limited by nice segmentations
        if "used_videos" in datadict:
            self.used_videos = datadict["used_videos"]
        else:
            self.used_videos = sorted(
                [name[:-4] for name in os.listdir(self.objectmasks)]
            )
        # Number of frames either given for all or read out based on the segmentation masks
        if "FRAMES_ALL_VIDS" in datadict:
            self.FRAMES_ALL_VIDS = datadict["FRAMES_ALL_VIDS"]
            self.video_frames = {
                video: self.FRAMES_ALL_VIDS for video in self.used_videos
            }
        else:
            self.video_frames = {
                video: np.load(f"{self.objectmasks}{video}.npy").shape[0]
                for video in self.used_videos
            }
        # get the video dimensions from the segmentation masks, if not provided
        if {"VID_SIZE_X", "VID_SIZE_Y"} <= set(datadict):
            self.VID_SIZE_X = datadict["self.VID_SIZE_X"]
            self.VID_SIZE_Y = datadict["self.VID_SIZE_Y"]
        else:
            masks = np.load(
                f"{self.objectmasks}{self.used_videos[0]}.npy"
            )
            if len(masks.shape) == 3:
                self.VID_SIZE_Y, self.VID_SIZE_X = masks.shape[1:]
            else:
                self.VID_SIZE_Y, self.VID_SIZE_X = masks.shape
        # For evaluation if gaze point is attributed to an object, defaults to 1 DVA
        if "RADIUS_OBJ_GAZE" in datadict:
            self.RADIUS_OBJ_GAZE = datadict["RADIUS_OBJ_GAZE"]
        else:
            self.RADIUS_OBJ_GAZE = 1.0 * self.DVA_TO_PX
        # check if predefined train or testset is specified
        if "trainset" in datadict:
            assert set(datadict["trainset"]) <= set(self.used_videos)
            self.trainset = datadict["trainset"]
        else:
            self.trainset = self.used_videos
        if "testset" in datadict:
            assert set(datadict["testset"]) <= set(self.used_videos)
            self.testset = datadict["testset"]
        else:
            self.testset = None
        # path for storing visualizations
        if "outputpath" in datadict:
            self.outputpath = datadict["outputpath"]

        self.video_col = datadict.get("NAME_COL", "video") 
        print('Video column is set to:', self.video_col)
        # Check if a path to the ground truth foveation evaluation dataframe is provided.
        # If not, there should be the path to the files and a function to do this evaluation in a class method.
        if "gt_foveation_df" in datadict:
            self.gt_foveation_df = pd.read_csv(
                self.PATH + datadict["gt_foveation_df"]
            )
            if self.trainset:
                self.train_foveation_df = self.gt_foveation_df[
                    self.gt_foveation_df[self.video_col].isin(self.trainset)
                ]
            else:
                self.train_foveation_df = pd.DataFrame()
            if self.testset:
                self.test_foveation_df = self.gt_foveation_df[
                    self.gt_foveation_df[self.video_col].isin(self.testset)
                ]
            else:
                self.test_foveation_df = pd.DataFrame()
        else:
            assert (
                "eye_tracking_data" in datadict
            ), f"gt_foveation_df or eye_tracking_data needed in dataconfig!"
            self.gt_foveation_df = self.create_foveation_df(
                datadict["eye_tracking_data"]
            )

        # for NSS evaluation, another df has to be provided...
        if "gt_fovframes_nss_df" in datadict:
            self.gt_fovframes_nss_df = pd.read_csv(
                self.PATH + datadict["gt_fovframes_nss_df"],
                usecols=["frame", "x", "y", "subject",  self.video_col, "nss"],
            )
        elif "eye_tracking_data" in datadict:
            self.gt_fovframes_nss_df = self.create_nss_df(datadict["eye_tracking_data"])
        else:
            self.gt_fovframes_nss_df = None

    def load_yaml(self, path):
        """
        Load a yaml file into a dict.

        :param path: Path to the yaml file.
        :type path: str
        :return: Dictionary with the yaml file content.
        :rtype: dict
        """
        with open(path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def load_objectmasks(self, videoname):
        """
        Loads the object masks for a given video. It assumes that these masks are stored in objectmasks/videoname as .npy file.

        :param videoname: Video for which the object masks are loaded
        :type videoname: str
        :return: Object segmentation masks of shape (frames, VID_SIZE_Y, VID_SIZE_X)
        :rtype: np.ndarray
        """
        masks = np.load(f"{self.objectmasks}{videoname}.npy")
        if len(masks.shape) == 2: # input is only 2D, broadcast to 3D
            broadcasted_mask = np.broadcast_to(masks, (self.video_frames[videoname], masks.shape[0], masks.shape[1]))
            masks = broadcasted_mask
        if masks.shape[1:] != (self.VID_SIZE_Y, self.VID_SIZE_X):
                masks_resized = np.zeros(
                    (masks.shape[0], self.VID_SIZE_Y, self.VID_SIZE_X), dtype=np.float32
                )
                for i in range(masks.shape[0]):
                    masks_resized[i] = resize(masks[i], (self.VID_SIZE_Y, self.VID_SIZE_X))
                masks = masks_resized
        assert masks.shape == (
            self.video_frames[videoname],
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
        ), f"Segmentation masks are not in shape (f,y,x)! Shape: {masks.shape}"
        return masks

    def load_featuremaps(self, videoname, featuretype=None, centerbias=None, scalerange=[0,0], normalized_std=0):
        """
        Loads the feature maps for a given video. It assumes that these masks are stored in featuremaps/featuretype/videoname as .npy file.

        :param videoname: Video for which the object masks are loaded
        :type videoname: str
        :param featuretype: Type of feature maps to load, has to be provided in a subdir, defaults to None
        :type featuretype: str, optional
        :param centerbias: Multiplicative center bias,if `anisotropic_default` it uses anisotropic_centerbias with default params.
                           Otherwise, it must be of shape (VID_SIZE_Y, VID_SIZE_X) and will be used directly, defaults to None
        :type centerbias: np.ndarray or str, optional
        :return: Feature maps of shape (frames, VID_SIZE_Y, VID_SIZE_X)
        :rtype: np.ndarray
        """
        if featuretype in set((None, "None")):
            featuremaps = 0.5 * np.ones(
                (self.video_frames[videoname], self.VID_SIZE_Y, self.VID_SIZE_X)
            )
        elif os.path.exists(f"{self.featuremaps}{featuretype}/{videoname}.npy"):
            featuremaps = np.load(f"{self.featuremaps}{featuretype}/{videoname}.npy")
        if scalerange != [0,0]:
            # scale maps such that min is scalerange[0] and max is scalerange[1]
            featuremaps = (featuremaps - np.min(featuremaps)) / (np.max(featuremaps) - np.min(featuremaps))
            featuremaps = featuremaps * (scalerange[1] - scalerange[0]) + scalerange[0]
        if normalized_std != 0:
            featuremaps = featuremaps.astype(np.float32)
            featuremaps = (featuremaps - np.mean(featuremaps)) / np.std(featuremaps) 
            featuremaps = np.clip(featuremaps * normalized_std + 1, 0, None)
            
        if featuremaps.ndim == 2:  # only (y, x)
            featuremaps = np.broadcast_to(
                featuremaps[np.newaxis, :, :],
                (self.video_frames[videoname], featuremaps.shape[0], featuremaps.shape[1])
            ).astype(np.float32)
        if featuremaps.ndim == 3:
             if featuremaps.shape[1:] != (self.VID_SIZE_Y, self.VID_SIZE_X):
                featuremaps_resized = np.zeros(
                    (featuremaps.shape[0], self.VID_SIZE_Y, self.VID_SIZE_X), dtype=np.float32
                )
                for i in range(featuremaps.shape[0]):
                    featuremaps_resized[i] = resize(featuremaps[i], (self.VID_SIZE_Y, self.VID_SIZE_X))
                featuremaps = featuremaps_resized
        print(f"VID_SIZE_Y={self.VID_SIZE_Y}, VID_SIZE_X={self.VID_SIZE_X}")
        print(f"Feature maps shape: {featuremaps.shape}")
        print(f"Expected frames: {self.video_frames[videoname]}")
        assert featuremaps.shape == (
            self.video_frames[videoname],
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
        ), "Feature maps are not in shape (f,y,x)!"
        if centerbias is None:
            return featuremaps
        elif centerbias == "anisotropic_default":
            return featuremaps * anisotropic_centerbias(
                self.VID_SIZE_X, self.VID_SIZE_Y
            )
        # elif: Implement other keywords?!
        else:
            assert centerbias.shape == (
                self.VID_SIZE_Y,
                self.VID_SIZE_X,
            ), "Provided center bias needs to match the size of the feature map!"
            return featuremaps * centerbias

    def load_topdownmaps(self, videoname, normalized_std):
        """
        Loads the top down maps for a given video. 
        It assumes that these masks are stored in topdownmaps/videoname as .npy file.

        :param videoname: Video for which the topp down maps are loaded
        :type videoname: str
        :return: Top down maps of shape (frames, VID_SIZE_Y, VID_SIZE_X)
        :rtype: np.ndarray
        """
        topdownmaps = np.load(f"{self.topdownmaps}{videoname}.npy").astype(np.float32)
        topdownmaps = (topdownmaps - np.mean(topdownmaps)) / np.std(topdownmaps) 
        topdownmaps = topdownmaps * normalized_std + 1
        topdownmaps = np.clip(topdownmaps, 0, None)
        
        if len(topdownmaps.shape) == 2: # input is only 2D, broadcast to 3D
            broadcasted_maps = np.broadcast_to(topdownmaps, (self.video_frames[videoname], topdownmaps.shape[0], topdownmaps.shape[1]))
            topdownmaps = broadcasted_maps

        assert topdownmaps.shape == (
            self.video_frames[videoname],
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
        ), "Top-down maps are not in shape (f,y,x)!"
        return topdownmaps

    def load_flowmaps(self, videoname):
        """
        Loads the optical flow maps for a given video, which have the shape (f-1,y,x,2).
        It assumes that these maps are stored in flowmaps/videoname as .npy file.

        :param videoname: Video for which the OF maps are loaded
        :type videoname: str
        :return: Optical flow maps of shape (frames-1, VID_SIZE_Y, VID_SIZE_X, 2)
        :rtype: np.ndarray
        """
        flowmaps = np.load(f"{self.flowmaps}{videoname}.npy").astype(np.float32)
        assert flowmaps.shape == (
            self.video_frames[videoname] - 1,
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
            2,
        ), "OF maps are not in shape (f-1,y,x,2)!"
        return flowmaps

    def load_nssmaps(self, videoname):
        """
        Loads the ground truth foveation map for a given video, normalized for calculating the NSS score

        :param videoname: Video for which the object masks are loaded
        :type videoname: str
        :return: NSS maps of shape (frames, VID_SIZE_Y, VID_SIZE_X)
        :rtype: np.ndarray
        """
        nssmaps = np.load(f"{self.nssmaps}{videoname}.npy")
        assert nssmaps.shape == (
            self.video_frames[videoname],
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
        ), "NSS maps are not in shape (f,y,x)!"
        return nssmaps

    def load_videoframes(self, videoname):
        """
        Loads the frames of a given video, only used for visualization.
        Assumes that imageio with ffmpeg is installed.

        :param videoname: Video for which the object masks are loaded
        :type videoname: str
        :param ext: File extension of the video, defaults to "mpg"
        :type videoname: str
        :return: NSS maps of shape (frames, VID_SIZE_Y, VID_SIZE_X)
        :rtype: np.ndarray
        """
        
        if self.vid_ext in ["png", "jpg", "jpeg"]:
            # load single png using imageio
            vid = imageio.imread(f"{self.videoframes}{videoname}.{self.vid_ext}")
            nframes = self.video_frames[videoname]
            vidlist = [vid] * nframes
        else:
            vid = imageio.get_reader(f"{self.videoframes}{videoname}.{self.vid_ext}", "ffmpeg")
            for image in vid.iter_data():
                vidlist.append(np.array(image))
            del vid
            nframes = self.video_frames[videoname]

            assert len(vidlist) >= nframes, "Number of frames is too small!"
            vidlist = vidlist[:nframes]

        for i in range(len(vidlist)):
            img = vidlist[i]
            img_resized = resize(img, (self.VID_SIZE_Y, self.VID_SIZE_X, 3), anti_aliasing=True)
            vidlist[i] = (img_resized * 255).astype(np.uint8)

        assert vidlist[0].shape == (
            self.VID_SIZE_Y,
            self.VID_SIZE_X,
            3,
        ), f"Frames are not in shape (y,x,3), Framesize: {vidlist[0].shape}"

        return vidlist

    def get_fovcat_ratio(self, videos="all"):
        """
        Calculates the ratio of time that the human scanpaths on the given videos spent in each foveation category.

        :param videos: Videos that should be considered (`all`, `train`, `test` or `sgl_vid`), defaults to "all"
        :type videos: str, optional
        :raises Exception: Invalid `videos`
        :return: Dictionary with keys ["B", "D", "I", "R"] and how much time is spent in each
        :rtype: dict
        """
        if videos == "all":
            df_gtfov = self.gt_foveation_df
        elif videos == "train":
            df_gtfov = self.train_foveation_df
        elif videos == "test":
            df_gtfov = self.test_foveation_df
        elif videos in self.used_videos:
            df_gtfov = self.gt_foveation_df[self.gt_foveation_df[self.video_col] == videos]
        else:
            df_gtfov = pd.DataFrame()
            raise Exception(
                f"fovcat can be calcd for `all`, `train`, `test` or `sgl_vid`, you ask for {videos}"
            )

        assert (
            len(df_gtfov) > 0
        ), "`df_gtfov` is empty, make sure that `videos` is a valid input and that its not empty (if test/train)"
        categories = ["B", "D", "I", "R"]
        ratios = {}
        full_dur = np.nansum(df_gtfov.duration_ms)
        #original ScanDy-Data with fov_category column, ScanDy-PFC-Data with fov_cat column
        if 'fov_category' in df_gtfov.columns:
            cat_col = 'fov_category'
        elif 'fov_cat' in df_gtfov.columns:
            cat_col = 'fov_cat'
        else:
            raise KeyError("Neither 'subject' nor 'sub_id' column found in df_temp")

        for cat in categories:
            ratio = (
                np.nansum(df_gtfov[df_gtfov[cat_col] == cat].duration_ms)
                / full_dur
            )
            ratios[cat] = ratio
        return ratios

    def get_foveation_ratio(self):
        """
        Returns the ratio of time spent during foveation across the dataset.
        In simulation, this is 1, since saccades are instantaneous and there is no tracker-noise or blinks.
        This affects the detection ratio, since all objects can only be detected once.
        """
        assert (
            len(self.gt_foveation_df) > 0
        ), "`result_df` is empty, make sure to run `evaluate_all_to_df` first!"
        fov_dur = np.sum(self.gt_foveation_df.duration_ms)
        full_dur = 0
        for vid in set(self.gt_foveation_df[self.video_col]): #was video earlier
            if vid not in self.video_frames:
                # skip videos that are not part of the dataset, makes it possible to run with less
                continue
            nframes = self.video_frames[vid]  # in case some videos are longer
            df_temp = self.gt_foveation_df[self.gt_foveation_df[self.video_col] == vid]
            #original ScanDy-Data with subject column, ScanDy-PFC-Data with subj-id column
            if 'subject' in df_temp.columns:
                subj_col = 'subject'
            elif 'subj_id' in df_temp.columns:
                subj_col = 'subj_id'
            else:
                raise KeyError("Neither 'subject' nor 'subj_id' column found in df_temp")

            for s in df_temp[subj_col].dropna().unique():
                            full_dur += nframes / self.FPS * 1000  # unit is ms!
        return fov_dur / full_dur


    #######################################################################
    ##########                NOT YET IMPLEMENTED                ##########
    #######################################################################

    def create_foveation_df(self, eye_tracking_data):
        """
        Will create a dataframe with all ground truth foveations and their statistics
        from the ground truth eye tracking data (what format though?).
        """
        raise NotImplementedError

    def create_nss_df(self, eye_tracking_data):
        """
        Will create a dataframe with all nss scores from the ground truth eye tracking data
        (what format though?).
        """
        raise NotImplementedError
