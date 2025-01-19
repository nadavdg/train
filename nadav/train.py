import argparse

import util


if __name__ == "__main__":
    """
    annotations should be provided in yolo format, this is: 
            class, xc, yc, w, h
    data needs to follow this structure:
    
    data-dir
    ----- train
    --------- imgs
    ------------ filename0001.jpg
    ------------ filename0002.jpg
    ------------ ....
    --------- anns
    ------------ filename0001.txt
    ------------ filename0002.txt
    ------------ ....
    ----- val
    --------- imgs
    ------------ filename0001.jpg
    ------------ filename0002.jpg
    ------------ ....
    --------- anns
    ------------ filename0001.txt
    ------------ filename0002.txt
    ------------ ....
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--class-list', default='/home/nadav/Downloads/task_day light oranges ldc-2025_01_16_08_47_16-coco 1.0/class.names')
    parser.add_argument('--data-dir', default='/home/nadav/Downloads/task_day light oranges ldc-2025_01_16_08_47_16-coco 1.0/data')
    parser.add_argument('--output-dir', default='/home/nadav/Downloads/task_day light oranges ldc-2025_01_16_08_47_16-coco 1.0/output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--learning-rate', default=0.00025)
    parser.add_argument('--batch-size', default=3)
    parser.add_argument('--iterations', default=10000)
    parser.add_argument('--checkpoint-period', default=500)
    parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml')

    args = parser.parse_args()

    util.train(args.output_dir,
               args.data_dir,
               args.class_list,
               device=args.device,
               learning_rate=float(args.learning_rate),
               batch_size=int(args.batch_size),
               iterations=int(args.iterations),
               checkpoint_period=int(args.checkpoint_period),
               model=args.model)
