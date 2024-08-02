from .dataset import ColorDataset, MNPZDataset, NPZDataset, PickleDataset, LanguageDataset, SynchronizedDatasets


def load_dataset_nerf(n_perspectives, path):
    color_dataset = ColorDataset(name='color', n_perspectives=n_perspectives,
                                 path=path)
    camera_config_dataset = MNPZDataset(keywords=['intrinsics', 'pose'], name='camera_config',
                                        path=path,
                                        n_samples=n_perspectives)
    info_dataset = PickleDataset(name='info', path=path)
    datasets = {
        'color': color_dataset,
        'camera_config': camera_config_dataset,
        'info': info_dataset,
    }
    synchronized_dataset = SynchronizedDatasets(datasets)
    print(
        f"Dataset {path} loaded with {len(synchronized_dataset)} samples")
    return synchronized_dataset


def store_to_dataset_nerf(synchronized_dataset, observations, info, seed):
    images = [observation['color'] for observation in observations]
    camera_configs = [{'intrinsics': observation['intrinsics'], 'pose': observation['pose']} for observation
                      in observations]
    data_sample = {
        'color': images,
        'camera_config': camera_configs,
        'info': info
    }
    synchronized_dataset.add_sample(seed, data_sample)


def load_dataset_goal(n_perspectives, path):
    color_dataset = ColorDataset(name='color', n_perspectives=n_perspectives,
                                 path=path)
    camera_config_dataset = MNPZDataset(keywords=['intrinsics', 'pose'], name='camera_config',
                                        path=path,
                                        n_samples=n_perspectives)
    grasp_pose_dataset = NPZDataset(
        keywords=['grasp_pose'], name='grasp_pose', path=path)
    info_dataset = PickleDataset(name='info', path=path)
    datasets = {'color': color_dataset,
                'camera_config': camera_config_dataset,
                'grasp_pose': grasp_pose_dataset,
                'info': info_dataset}
    synchronized_dataset = SynchronizedDatasets(datasets)
    print(f"Dataset {path} loaded with {len(synchronized_dataset)} samples")
    return synchronized_dataset


def store_to_dataset_goal(synchronized_dataset, observations, grasp_pose, info, seed):
    images = [observation['color'] for observation in observations]
    camera_configs = [{'intrinsics': observation['intrinsics'], 'pose': observation['pose']} for observation
                      in observations]
    data_sample = {
        'color': images,
        'camera_config': camera_configs,
        'grasp_pose': {'grasp_pose': grasp_pose},
        'info': info
    }
    synchronized_dataset.add_sample(seed, data_sample)


def load_dataset_trajectory(n_perspectives, path):
    color_dataset = ColorDataset(name='color', n_perspectives=n_perspectives,
                                 path=path)
    camera_config_dataset = MNPZDataset(keywords=['intrinsics', 'pose'], name='camera_config',
                                        path=path,
                                        n_samples=n_perspectives)
    grasp_pose_dataset = NPZDataset(
        keywords=['grasp_pose'], name='grasp_pose', path=path)
    info_dataset = PickleDataset(name='info', path=path)
    trajectory_dataset = NPZDataset(keywords=['trajectory'], name='trajectory',
                                    path=path)
    datasets = {'color': color_dataset,
                'camera_config': camera_config_dataset,
                'grasp_pose': grasp_pose_dataset,
                'info': info_dataset,
                'trajectory': trajectory_dataset}
    synchronized_dataset = SynchronizedDatasets(datasets)
    print(f"Dataset {path} loaded with {len(synchronized_dataset)} samples")
    return synchronized_dataset


def store_to_dataset_trajectory(synchronized_dataset, observations, steps, grasp_pose, info, seed):
    images = [observation['color'] for observation in observations]
    camera_configs = [{'intrinsics': observation['intrinsics'], 'pose': observation['pose']} for observation
                      in observations]
    data_sample = {
        'color': images,
        'camera_config': camera_configs,
        'grasp_pose': {'grasp_pose': grasp_pose},
        'trajectory': {'trajectory': steps},
        'info': info
    }
    synchronized_dataset.add_sample(seed, data_sample)


def load_dataset_language(n_perspectives, path):
    color_dataset = ColorDataset(name='color', n_perspectives=n_perspectives,
                                 path=path)
    camera_config_dataset = MNPZDataset(keywords=['intrinsics', 'pose'], name='camera_config',
                                        path=path,
                                        n_samples=n_perspectives)
    grasp_pose_dataset = NPZDataset(
        keywords=['grasp_pose'], name='grasp_pose', path=path)
    info_dataset = PickleDataset(name='info', path=path)
    trajectory_dataset = NPZDataset(keywords=['trajectory'], name='trajectory',
                                    path=path)
    language_dataset = LanguageDataset(name='language', path=path)

    datasets = {'color': color_dataset,
                'camera_config': camera_config_dataset,
                'grasp_pose': grasp_pose_dataset,
                'info': info_dataset,
                'trajectory': trajectory_dataset,
                'language': language_dataset}
    synchronized_dataset = SynchronizedDatasets(datasets)
    print(f"Dataset {path} loaded with {len(synchronized_dataset)} samples")
    return synchronized_dataset


def store_to_dataset_language(synchronized_dataset, observations, steps, grasp_pose, info, language, seed):
    images = [observation['color'] for observation in observations]
    camera_configs = [{'intrinsics': observation['intrinsics'], 'pose': observation['pose']} for observation
                      in observations]
    data_sample = {
        'color': images,
        'camera_config': camera_configs,
        'grasp_pose': {'grasp_pose': grasp_pose},
        'trajectory': {'trajectory': steps},
        'info': info,
        'language': language
    }
    synchronized_dataset.add_sample(seed, data_sample)
