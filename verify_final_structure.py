import logging
from src.data.dataset import get_dataset
from src.data.split import Partitions
from src.config import AllConfig, Experiment, hydra_main
from src.config.options import Datasets

logging.basicConfig(level=logging.INFO)


def verify_dataset(dataset_name):
    logging.info(f'Checking {dataset_name}...')

    cfg = Experiment.get_config()
    cfg.data.dataset.name = dataset_name
    cfg.data.n_input_points = 2048

    # Use a small subset for quick verification
    if dataset_name == Datasets.ShapeNetFlow:
        cfg.data.dataset.selected_classes = ['airplane']
    elif dataset_name == Datasets.ModelNet:
        cfg.data.dataset.selected_classes = ['airplane']

    try:
        ds_split = get_dataset(Partitions.train)
        logging.info(f'Successfully loaded {dataset_name} train split. Size: {len(ds_split)}')

        sample_inputs, sample_targets = ds_split[0]

        logging.info(f'Sample cloud shape: {sample_inputs.cloud.shape}')
        logging.info(f'Sample cloud_512 shape: {sample_inputs.cloud_512.shape}')
        logging.info(f'Sample cloud_128 shape: {sample_inputs.cloud_128.shape}')
        logging.info(f'Sample label: {sample_targets.label}')

        assert hasattr(sample_inputs, 'initial_sampling')
        assert sample_inputs.cloud.shape == (2048, 3)
        assert sample_inputs.cloud_512.shape == (512, 3)
        assert sample_inputs.cloud_128.shape == (128, 3)

        # Check val split for ModelNet specifically to verify the new preprocessing logic
        if dataset_name == Datasets.ModelNet:
            val_ds = get_dataset(Partitions.val)
            logging.info(f'ModelNet val split size: {len(val_ds)}')
            assert len(val_ds) > 0

        logging.info(f'{dataset_name} verification PASSED')
    except Exception as e:
        logging.error(f'{dataset_name} verification FAILED: {e}')
        import traceback

        traceback.print_exc()


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name='verification', par_dir=cfg.user.path.version_dir)
    with exp.create_run():
        # Test ModelNet
        verify_dataset(Datasets.ModelNet)

        # Test ShapeNet
        verify_dataset(Datasets.ShapeNetFlow)


if __name__ == '__main__':
    main()
