import os

import rastervision as rv


def build_scene(task, data_uri, id, channel_order=None):
    id = id.replace('-', '_')
    raster_source_uri = '{}/isprs-potsdam/4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(data_uri, id)
    #os.path.isfile("
    label_source_uri = '{}/labels/all/top_potsdam_{}_RGBIR.json'.format(data_uri, id)

    # Using with_rgb_class_map because input TIFFs have classes encoded as RGB colors.
    label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
		 .with_uri(label_source_uri) \
                 .with_ioa_thresh(0.5) \
                 .with_use_intersection_over_cell(False) \
                 .with_pick_min_class_id(True) \
                 .with_background_class_id(2) \
                 .with_infer_cells(True) \
                 .build()
    # URI will be injected by scene config.
    # Using with_rgb(True) because we want prediction TIFFs to be in RGB format.
    scene = rv.SceneConfig.builder() \
                          .with_task(task) \
                          .with_id(id) \
                          .with_raster_source(raster_source_uri,
                                              channel_order=channel_order) \
                          .with_label_source(label_source) \
                          .build()

    return scene


class PotsdamChipClassification(rv.ExperimentSet):
    def exp_main(self, root_uri, data_uri, test_run=False):
        """Run an experiment on the ISPRS Potsdam dataset.

        Uses Tensorflow Deeplab backend with Mobilenet architecture. Should get to
        F1 score of ~0.86 (including clutter class) after 6 hours of training on P3
        instance.

        Args:
            root_uri: (str) root directory for experiment output
            data_uri: (str) root directory of Potsdam dataset
            test_run: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        if test_run == 'True':
            test_run = True
        elif test_run == 'False':
            test_run = False

        train_ids = ['2_10', '2_11', '2_12', '2_14', '3_11',
                           '3_13', '4_10', '5_10', '6_7', '6_9']
        val_ids = ['2_13', '6_8', '3_10']
        # infrared, red, green
        channel_order = [3, 0, 1]

        debug = False
        batch_size = 8
        num_epochs = 40

        # Better results can be obtained at a greater computational expense using
        # num_steps = 150000
        # model_type = rv.XCEPTION_65

        if test_run:
            debug = True
            num_epochs = 1
            batch_size = 1
            train_ids = train_ids[0:1]
            val_ids = val_ids[0:1]

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
			.with_chip_size(200) \
			.with_classes({
				"car": (1, "red"),
				"no_car": (2, "black")
			}) \
			.build()

        backend = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                  .with_task(task) \
                                  .with_model_defaults(rv.RESNET50_IMAGENET) \
                                  .with_debug(debug) \
                                  .with_train_options(replace_model=True) \
                                  .with_batch_size(batch_size) \
                                  .with_num_epochs(num_epochs) \
                                  .with_config({
                                      "trainer": {
                                          "options": {
                                              "saveBest": True,
                                              "lrSchedule": [
                                                  {
                                                      "epoch": 0,
                                                      "lr": 0.0005
                                                  },
                                                  {
                                                      "epoch": 15,
                                                      "lr": 0.0001
                                                  },
                                                  {
                                                      "epoch": 30,
                                                      "lr": 0.00001
                                                  }
                                              ]
                                          }
                                      }
                                  }, set_missing_keys=True) \
                                  .build()

        train_scenes = [build_scene(task, data_uri, id, channel_order)
                        for id in train_ids]
        val_scenes = [build_scene(task, data_uri, id, channel_order)
                      for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('potsdam_chip') \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
