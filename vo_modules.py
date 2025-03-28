from sklearn import linear_model
from time import time
from tqdm import tqdm
from libs.general.timer import Timers

from libs.deep_depth.monodepth2 import Monodepth2DepthNet
from libs.geometry.ops_3d import *

from libs.camera_modules import SE3, Intrinsics
from libs.utils import *


class VisualOdometry():
    def __init__(self, cfg):
        """
        Args:
            cfg (edict): configuration reading from yaml file
        """
        # camera intrinsics
        self.cam_intrinsics = Intrinsics()

        # predicted global poses
        self.global_poses = {0: SE3()}

        # configuration
        self.cfg = cfg

        # timer
        self.timers = Timers()
        self.timers.add(["Depth-CNN",
                         "Ess. Mat.",
                         "visualization",
                         "visualization_depth" ])




    def get_intrinsics_param(self, dataset):
        """Read intrinsics parameters for each dataset
        Args:
            dataset (str): dataset
                - kitti
                - tum-1/2/3
        Returns:
            intrinsics_param (float list): [cx, cy, fx, fy]
        """
        # Kitti
        if dataset == "kitti":
            img_seq_dir = os.path.join(
                self.cfg.directory.img_seq_dir,
                self.cfg.seq
            )
            intrinsics_param = load_kitti_odom_intrinsics(
                os.path.join(img_seq_dir, "calib.txt")
            )[2]
        # TUM
        elif "tum" in dataset:
            tum_intrinsics = {
                "tum-1": [318.6, 255.3, 517.3, 516.5],  # fr1
                "tum-2": [325.1, 249.7, 520.9, 521.0],  # fr2
                "tum-3": [320.1, 247.6, 535.4, 539.2],  # fr3
            }
            intrinsics_param = tum_intrinsics[dataset]
        return intrinsics_param



    def get_img_depth_dir(self):
        """Get image data directory and (optional) depth data directory

        Returns:
            img_data_dir (str): image data directory
            depth_data_dir (str): depth data directory / None
            depth_src (str): depth data type
                - gt
                - None
        """
        # get image data directory
        img_seq_dir = os.path.join(
            self.cfg.directory.img_seq_dir,
            self.cfg.seq
        )
        if self.cfg.dataset == "kitti":
            img_data_dir = os.path.join(img_seq_dir, "image_2")
        elif "tum" in self.cfg.dataset:
            img_data_dir = os.path.join(img_seq_dir, "rgb")
        else:
            warn_msg = "Wrong dataset [{}] is given.".format(self.cfg.dataset)
            warn_msg += "\n Choose from [kitti, tum-1/2/3]"
            assert False, warn_msg

        # get depth data directory
        depth_src_cases = {
            0: "gt",
            None: None
        }
        depth_src = depth_src_cases[self.cfg.depth.depth_src]

        if self.cfg.dataset == "kitti":
            if depth_src == "gt":
                depth_data_dir = "{}/gt/{}/".format(
                    self.cfg.directory.depth_dir, self.cfg.seq
                )
            elif depth_src is None:
                depth_data_dir = None
        elif "tum" in self.cfg.dataset:
            if depth_src == "gt":
                depth_data_dir = "{}/{}/depth".format(
                    self.cfg.directory.depth_dir, self.cfg.seq
                )
            elif depth_src is None:
                depth_data_dir = None

        return img_data_dir, depth_data_dir, depth_src


    def initialize_deep_depth_model(self):
        """Initialize single-view depth model
        Returns:
            depth_net: single-view depth network
        """
        depth_net = Monodepth2DepthNet()
        depth_net.initialize_network_model(
            weight_path=self.cfg.depth.pretrained_model,
            dataset=self.cfg.dataset)
        return depth_net

    def get_gt_poses(self):
        """load ground-truth poses
        Returns:
            gt_poses (dict): each pose is 4x4 array
        """
        if self.cfg.directory.gt_pose_dir is not None:
            if self.cfg.dataset == "kitti":
                annotations = os.path.join(
                    self.cfg.directory.gt_pose_dir,
                    "{}.txt".format(self.cfg.seq)
                )
                gt_poses = load_poses_from_txt(annotations)
            elif "tum" in self.cfg.dataset:
                annotations = os.path.join(
                    self.cfg.directory.gt_pose_dir,
                    self.cfg.seq,
                    "groundtruth.txt"
                )
                gt_poses = load_poses_from_txt_tum(annotations)
        return gt_poses

    def setup(self):
        """Reading configuration and setup, including
        - Get camera intrinsics
        - Get tracking method
        - Get feature tracking method
        - Get image & (optional depth) data
        - Generate keypoint sampling scheme
        - Deep networks
        - Load GT poses
        - Set drawer
        """
        # read camera intrinsics
        intrinsics_param = self.get_intrinsics_param(self.cfg.dataset)
        self.cam_intrinsics = Intrinsics(intrinsics_param)

        # get image and depth data directory
        self.img_path_dir, self.depth_seq_dir, self.depth_src = self.get_img_depth_dir()

        # generate keypoint sampling scheme

        # Deep networks
        self.deep_models = {}
        # optical flow

        # single-view depth
        if self.depth_src is None:
            if self.cfg.depth.pretrained_model is not None:
                self.deep_models['depth'] = self.initialize_deep_depth_model()
            else:
                assert False, "No precomputed depths nor pretrained depth model"

        # Load GT pose
        self.gt_poses = self.get_gt_poses()

        # Set drawer

    def compute_pose_3d2d(self, kp1, kp2, depth_1):
        """Compute pose from 3d-2d correspondences
        Args:
            kp1 (Nx2 array): keypoints for view-1
            kp2 (Nx2 array): keypoints for view-2
            depth_1 (HxW array): depths for view-1
        Returns:
            pose (SE3): relative pose from view-2 to view-1
            kp1 (Nx2 array): filtered keypoints for view-1
            kp2 (Nx2 array): filtered keypoints for view-2
        """
        height, width = depth_1.shape

        # Filter keypoints outside image region
        x_idx = (kp2[:, 0] >= 0) * (kp2[:, 0] < width)
        kp1 = kp1[x_idx]
        kp2 = kp2[x_idx]
        y_idx = (kp2[:, 1] >= 0) * (kp2[:, 1] < height)
        kp1 = kp1[y_idx]
        kp2 = kp2[y_idx]

        # Filter keypoints outside depth range
        kp1_int = kp1.astype(np.int)
        kp_depths = depth_1[kp1_int[:, 1], kp1_int[:, 0]]
        non_zero_mask = (kp_depths != 0)
        depth_range_mask = (kp_depths < self.cfg.depth.max_depth) * (kp_depths > self.cfg.depth.min_depth)
        valid_kp_mask = non_zero_mask * depth_range_mask

        kp1 = kp1[valid_kp_mask]
        kp2 = kp2[valid_kp_mask]

        # Get 3D coordinates for kp1
        XYZ_kp1 = unprojection_kp(kp1, kp_depths[valid_kp_mask], self.cam_intrinsics)

        # initialize ransac setup
        best_rt = []
        best_inlier = 0

        new_list = np.arange(0, kp2.shape[0], 1)
        np.random.shuffle(new_list)
        new_XYZ = XYZ_kp1.copy()[new_list]
        new_kp2 = kp2.copy()[new_list]
        if new_kp2.shape[0] > 4:
            flag, r, t, inlier = cv2.solvePnPRansac(
                objectPoints=new_XYZ,
                imagePoints=new_kp2,
                cameraMatrix=self.cam_intrinsics.mat,
                distCoeffs=None,
                iterationsCount=self.cfg.PnP.ransac.iter,
                reprojectionError=self.cfg.PnP.ransac.reproj_thre,
            )
            if flag and inlier.shape[0] > best_inlier:
                best_rt = [r, t]
                best_inlier = inlier.shape[0]
        pose = SE3()
        if len(best_rt) != 0:
            r, t = best_rt
            pose.R = cv2.Rodrigues(r)[0]
            pose.t = t
        pose.pose = pose.inv_pose
        return pose, kp1, kp2

    def find_scale_from_depth(self, kp1, kp2, T_21, depth2):
        """Compute VO scaling factor for T_21
        Args:
            kp1 (Nx2 array): reference kp
            kp2 (Nx2 array): current kp
            T_21 (4x4 array): relative pose; from view 1 to view 2
            depth2 (HxW array): depth 2
        Returns:
            scale (float): scaling factor
        """
        # Triangulation
        img_h, img_w, _ = image_shape(depth2)
        kp1_norm = kp1.copy()
        kp2_norm = kp2.copy()

        kp1_norm[:, 0] = \
            (kp1[:, 0] - self.cam_intrinsics.cx) / self.cam_intrinsics.fx
        kp1_norm[:, 1] = \
            (kp1[:, 1] - self.cam_intrinsics.cy) / self.cam_intrinsics.fy
        kp2_norm[:, 0] = \
            (kp2[:, 0] - self.cam_intrinsics.cx) / self.cam_intrinsics.fx
        kp2_norm[:, 1] = \
            (kp2[:, 1] - self.cam_intrinsics.cy) / self.cam_intrinsics.fy

        _, _, X2_tri = triangulation(kp1_norm, kp2_norm, np.eye(4), T_21)

        # Triangulation outlier removal
        depth2_tri = convert_sparse3D_to_depth(kp2, X2_tri, img_h, img_w)
        depth2_tri[depth2_tri < 0] = 0

        # common mask filtering
        non_zero_mask_pred = (depth2 > 0)
        non_zero_mask_tri = (depth2_tri > 0)
        valid_mask = non_zero_mask_pred * non_zero_mask_tri
        depth_pred_non_zero = np.concatenate([depth2[valid_mask]])
        depth_tri_non_zero = np.concatenate([depth2_tri[valid_mask]])
        depth_ratio = depth_tri_non_zero / depth_pred_non_zero


        # # Estimate scale (ransac)
        if valid_mask.sum() > 10:  # self.cfg.translation_scale.ransac.min_samples:
            # RANSAC scaling solver
            ransac = linear_model.RANSACRegressor(
                base_estimator=linear_model.LinearRegression(
                    fit_intercept=False),
                min_samples=self.cfg.translation_scale.ransac.min_samples,
                max_trials=self.cfg.translation_scale.ransac.max_trials,
                stop_probability=self.cfg.translation_scale.ransac.stop_prob,
                residual_threshold=self.cfg.translation_scale.ransac.thre
            )
            ransac.fit(
                depth_ratio.reshape(-1, 1),
                np.ones((depth_ratio.shape[0], 1))
            )
            scale = ransac.estimator_.coef_[0, 0]
        else:
            scale = -100

        return scale
    def compute_pose_2d2d(self, kp_ref, kp_cur):
        """Compute the pose from view2 to view1
        Args:
            kp_ref (Nx2 array): keypoints for reference view
            kp_cur (Nx2 array): keypoints for current view
        Returns:
            pose (SE3): relative pose from current to reference view
            best_inliers (N boolean array): inlier mask
        """
        principal_points = (self.cam_intrinsics.cx, self.cam_intrinsics.cy)


        # initialize ransac setup
        best_Rt = []
        best_inlier_cnt = 0
        # max_ransac_iter = self.cfg.compute_2d2d_pose.ransac.repeat
        best_inliers = np.ones((kp_ref.shape[0])) == 1

        # check flow magnitude
        avg_flow = np.mean(np.linalg.norm(kp_ref-kp_cur, axis=1))
        min_flow = self.cfg.compute_2d2d_pose.min_flow
        if avg_flow > min_flow:
            new_list = np.random.randint(0, kp_cur.shape[0], (kp_cur.shape[0]))
            new_kp_cur = kp_cur.copy()[new_list]
            new_kp_ref = kp_ref.copy()[new_list]
            start_time = time()
            E, inliers = cv2.findEssentialMat(
                new_kp_cur,
                new_kp_ref,
                focal=self.cam_intrinsics.fx,
                pp=principal_points,
                method=cv2.RANSAC,
                prob=0.99,
                threshold=self.cfg.compute_2d2d_pose.ransac.reproj_thre,
            )

            cheirality_cnt, R, t, _ = cv2.recoverPose(E, new_kp_cur, new_kp_ref, self.cam_intrinsics.mat, cv2.RANSAC)
            self.timers.timers["Ess. Mat."].append(time() - start_time)
            if inliers.sum() > best_inlier_cnt and cheirality_cnt > kp_cur.shape[0]*0.05:
                best_Rt = [R, t]
                best_inlier_cnt = inliers.sum()
                best_inliers = inliers

            if len(best_Rt) == 0:
                R = np.eye(3)
                t = np.zeros((3,1))
                best_Rt = [R, t]
        else:
            R = np.eye(3)
            t = np.zeros((3,1))
            best_Rt = [R, t]
        R, t = best_Rt
        pose = SE3()
        pose.R = R
        pose.t = t
        return pose, best_inliers

    def update_global_pose(self, new_pose, scale, img_id):
        """update estimated poses w.r.t global coordinate system
        Args:
            new_pose (SE3)
            scale (float): scaling factor
        """
        pose = SE3()

        pose.t = self.global_poses[img_id - 1].R @ new_pose.t * scale + self.global_poses[img_id - 1].t
        pose.R = self.global_poses[img_id - 1].R @ new_pose.R
        return pose

    def main(self):
        len_seq = len(self.gt_poses)
        print("==> Start VO")
        main_start_time = time()
        start_frame = 1

        traj = np.zeros((900, 1100, 3), dtype=np.uint8)

        for img_id in tqdm(range(start_frame, len_seq)):
            image1_color = read_image(self.img_path_dir + "/{:06d}.png".format(img_id - 1))
            image2_color = read_image(self.img_path_dir + "/{:06d}.png".format(img_id))
            img_h, img_w, _ = image_shape(image1_color)
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(image1_color, None)
            keypoints2, descriptors2 = sift.detectAndCompute(image2_color, None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            # flann = cv2.FlannBasedMatcher()
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
            vmatches = sorted(good_matches, key=lambda x: x.distance)  # 按距离升序排序
            visualmatches = vmatches[:50]  # 选择前50个较好的匹配
            i1_kp = [keypoints1[match.queryIdx].pt for match in good_matches]
            i2_kp = [keypoints2[match.trainIdx].pt for match in good_matches]
            image2_kp = np.array(i2_kp, dtype=np.float32)
            image1_kp = np.array(i1_kp, dtype=np.float32)
            # ransac_reproj_threshold = 200.0  # 距离阈值，可以根据实际情况调整
            #
            # # 提取匹配点的坐标
            # src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            #
            # # 使用 RANSAC 进行几何校验
            # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
            #
            # # 根据校验结果过滤匹配
            # filtered_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
            # i1_kp = [keypoints1[match.queryIdx].pt for match in filtered_matches]
            # i2_kp = [keypoints2[match.trainIdx].pt for match in filtered_matches]
            # image2_kp = np.array(i2_kp, dtype=np.float32)
            # image1_kp = np.array(i1_kp, dtype=np.float32)
            E_pose, _ = self.compute_pose_2d2d(image1_kp,image2_kp)
            posesave = SE3()
            deep_models = {}
            cur_data = {}
            ref_data = {}
            deep_models['depth'] = self.initialize_deep_depth_model()
            cur_data['raw_depth'] = \
                deep_models['depth'].inference(image2_color)
            cur_data['raw_depth'] = cv2.resize(cur_data['raw_depth'],
                                               (img_w, img_h),
                                               interpolation=cv2.INTER_NEAREST
                                               )
            ref_data['raw_depth'] = \
                deep_models['depth'].inference(image1_color)
            ref_data['raw_depth'] = cv2.resize(ref_data['raw_depth'],
                                               (img_w, img_h),
                                               interpolation=cv2.INTER_NEAREST
                                               )
            cur_data['depth'] = preprocess_depth(cur_data['raw_depth'], self.cfg.crop.depth_crop,
                                                 [self.cfg.depth.min_depth, self.cfg.depth.max_depth])
            cur_data['vdepth'] = preprocess_depth(cur_data['raw_depth'], [[0, 1], [0, 1]],
                                                  [self.cfg.depth.min_depth, self.cfg.depth.max_depth])
            ref_data['vdepth'] = preprocess_depth(ref_data['raw_depth'], [[0, 1], [0, 1]],
                                                  [self.cfg.depth.min_depth, self.cfg.depth.max_depth])
            ref_data['depth'] = preprocess_depth(ref_data['raw_depth'], self.cfg.crop.depth_crop,
                                                 [self.cfg.depth.min_depth, self.cfg.depth.max_depth])
            scale = self.find_scale_from_depth(
                image1_kp, image2_kp,
                E_pose.inv_pose, cur_data['depth']
            )
            if scale != -100:
                posesave.t = E_pose.t * scale
                posesave.R = E_pose.R
                posesave = self.update_global_pose(posesave, 1, img_id)

            else:
                # posesave.t = E_pose.t * 1
                # posesave.R = E_pose.R
                # posesave = self.update_global_pose(posesave, 1, img_id)
                pnp_pose, _, _ = self.compute_pose_3d2d(image1_kp, image2_kp, ref_data['depth'])
                posesave.t = self.global_poses[img_id-1].t + self.global_poses[img_id-1].R @ pnp_pose.t * 1
                posesave.R = pnp_pose.R @ self.global_poses[img_id-1].R
            self.global_poses[img_id] = posesave

            # 从旋转矩阵中计算欧拉角
            x, y, z = posesave.t[0], posesave.t[1], posesave.t[2]
            sy = math.sqrt(posesave.R[0, 0] * posesave.R[0, 0] + posesave.R[1, 0] * posesave.R[1, 0])
            roll = math.atan2(posesave.R[2, 1], posesave.R[2, 2])
            pitch = math.atan2(-posesave.R[2, 0], sy)
            yaw = math.atan2(posesave.R[1, 0], posesave.R[0, 0])
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)
            if self.cfg.dataset == "kitti":
                annotations = os.path.join(
                    self.cfg.directory.gt_pose_dir,
                    "{}.txt".format(self.cfg.seq)
                )
            with open(annotations) as f:
                tri = f.readlines()
            ss = tri[img_id].strip().split()
            trueX = float(ss[3])
            trueY = float(ss[7])
            trueZ = float(ss[11])

            # 在运动轨迹上添加当前相机位置的画点
            #00 376 1241
            draw_x, draw_y = int(x) + 660, ((-1) * (int(z))) + 660
            true_x, true_y = int(trueX) + 660, ((-1) * (int(trueZ))) + 660
            # 01 376 1241
            # draw_x, draw_y = int(x)-900, ((-1) * (int(z)))-500
            # true_x, true_y = int(trueX)-900, ((-1) * (int(trueZ)))-500
            # 02 376 1241
            # draw_x, draw_y = int(x) + 200, ((-1) * (int(z))) + 920
            # true_x, true_y = int(trueX) + 200, ((-1) * (int(trueZ))) + 920
            # 03 375 1242
            # draw_x, draw_y = int(x) + 70, ((-1) * (int(z))) + 480
            # true_x, true_y = int(trueX) + 70, ((-1) * (int(trueZ))) + 480
            # 04 370 1226
            # draw_x, draw_y = int(x) + 450, ((-1) * (int(z))) + 830
            # true_x, true_y = int(trueX) + 450, ((-1) * (int(trueZ))) + 830
            #05 370 1226
            draw_x, draw_y = int(x) + 400, ((-1) * (int(z))) + 660
            true_x, true_y = int(trueX) + 400, ((-1) * (int(trueZ))) + 660
            # 06 370 1226
            # draw_x, draw_y = int(x) + 450, ((-1) * (int(z))) + 630
            # true_x, true_y = int(trueX) + 450, ((-1) * (int(trueZ))) + 630

            #07 1226 370
            # draw_x, draw_y = int(x) + 450, ((-1) * (int(z))) + 450
            # true_x, true_y = int(trueX) + 450, ((-1) * (int(trueZ))) + 450
            # 08 370 1226
            # draw_x, draw_y = int(x) + 500, ((-1) * (int(z))) + 660
            # true_x, true_y = int(trueX) + 500, ((-1) * (int(trueZ))) + 660
            # 09 370 1226
            # draw_x, draw_y = int(x) + 300, ((-1) * (int(z))) + 700
            # true_x, true_y = int(trueX) + 300, ((-1) * (int(trueZ))) + 700
            #10 370 1226
            # draw_x, draw_y = int(x) + 10, ((-1) * (int(z))) + 450
            # true_x, true_y = int(trueX) + 10, ((-1) * (int(trueZ))) + 450


            cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)

            cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 2)
            cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)

            # 输出相机位置和旋转信息的文本
            text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
            textRot = "rotation: roll=%2fDeg pitch=%2fDeg yaw=%2fDeg" % (roll, pitch, yaw)

            # 将位置和旋转信息的文本显示在运动轨迹上
            cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
            cv2.putText(traj, textRot, (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
            image = cv2.resize(image1_color, (900, 300))  # 指定目标尺寸
            # 绘制匹配结果
            output_image = cv2.drawMatches(image1_color, keypoints1, image2_color, keypoints2, visualmatches, None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            resized_image = cv2.resize(output_image, (900, 300))  # 指定目标尺寸
            canvas = np.zeros((900, 2000, 3), dtype=np.uint8)

            # 分割左右两边
            left_half = canvas[:, :1100, :]
            right_half = canvas[:, 1100:2000, :]

            # 在左半边显示traj图像
            left_half[:traj.shape[0], :traj.shape[1], :] = traj  # traj是一个NumPy数组

            # 在右半边分为上中下三个部分
            image_height = right_half.shape[0] // 3

            # 显示image图像在上部
            right_half[:image_height, :image.shape[1], :] = image  # image是一个NumPy数组

            # 显示flowmatch图像在中部
            right_half[image_height:2 * image_height, :resized_image.shape[1], :] = resized_image  # flowmatch是一个NumPy数组

            # # 显示deoth图像在下部
            #
            # # 假设 cur_data['depth'] 是一个二维数组，表示深度图像
            # depth_image = cur_data['vdepth']
            #
            # # 将深度值映射到可视化颜色（这里使用的是伪彩色）
            # visualization = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255.0 / depth_image.max()),
            #                                   cv2.COLORMAP_JET)
            #
            # # 调整深度图像大小为900x300
            # resized_image = cv2.resize(visualization, (900, 300), interpolation=cv2.INTER_NEAREST)
            #
            # right_half[2 * image_height:, :cur_data['depth'].shape[1], :] = resized_image  # depth是一个NumPy数组

            # 使用OpenCV显示画布
            # 将深度图像1映射到可视化颜色（这里使用的是伪彩色）
            depth_image1 = ref_data['vdepth']
            depth_image2 = cur_data['vdepth']
            visualization1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image1, alpha=255.0 / depth_image1.max()),
                                               cv2.COLORMAP_JET)

            # 调整深度图像1大小为450x300
            resized_image1 = cv2.resize(visualization1, (450, 300), interpolation=cv2.INTER_NEAREST)

            # 将深度图像2映射到可视化颜色（这里使用的是伪彩色）
            visualization2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image2, alpha=255.0 / depth_image2.max()),
                                               cv2.COLORMAP_JET)

            # 调整深度图像2大小为450x300
            resized_image2 = cv2.resize(visualization2, (450, 300), interpolation=cv2.INTER_NEAREST)

            # 在右半边的下部分分为左右两个部分
            right_half[2 * image_height:, :450, :] = resized_image1  # depth_image1是一个NumPy数组
            right_half[2 * image_height:, 450:900, :] = resized_image2  # depth_image2是一个NumPy数组

            cv2.imshow("Canvas", canvas)




            # 每10ms刷新一次显示
            cv2.waitKey(1)



        print("=> Finish!")
        """ Display & Save result """
        # Output experiement information
        print("---- time breakdown ----")
        print("total runtime: {}".format(time() - main_start_time))

        # Save trajectory map
        print("Save VO map.")
        map_png = "{}/map{}.png".format(self.cfg.result_dir, self.cfg.seq)
        cv2.imwrite(map_png, canvas)

        # Save trajectory txt
        traj_txt = "{}/{}.txt".format(self.cfg.result_dir, self.cfg.seq)

        global_poses_arr = convert_SE3_to_arr(self.global_poses)
        save_traj(traj_txt, global_poses_arr, format="kitti")
