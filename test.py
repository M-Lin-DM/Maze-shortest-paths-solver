from maze_solving import EikonalSolver, PathFinder
from plotting import plot_optimal_path, plot_grayscale_img


def main():
    img_folder = "C:/Users/MrLin/OneDrive/Documents/Experiments/Eikonal_solver/images/class_0"
    # img_name = "rose2_opened.png"
    # img_name = "flowersmall.png"
    # img_name = "rose_erode_2x.png"
    # img_name = "manhattan_crop_dilated_flip.png"
    # img_name = "maze_png.png"
    img_name = "holy_moon_lvl_2.png"

    img_path = f"{img_folder}/{img_name}"

    # set parameters for distance field computation. Set start and end point by clicking in the figure.
    eikonal_solver = EikonalSolver(img_path, 2500,
                                   use_gpu=False)  # setting use_gpu=True led to a ~20x speedup on some images. Make sure pytorch, cudatoolkit, and cudnn are installed!

    # compute distance field from end point clicked. varbs holds the gradient of the distance field and other parameters used in PathFinder.
    T, varbs = eikonal_solver.find_distance_field()

    # plot_grayscale_img(T)  # plot distance field

    # Set parameters of path finding algorithm. eta represents the step size.
    path_finder = PathFinder(varbs, N_steps=2100, eta=0.5)

    # Output will be an array where each row is the x-y position at that step in the path.
    traj = path_finder.find_path()

    # plot contour plot of T and shortest path traj
    plot_optimal_path(T, traj, line_color="white")


if __name__ == "__main__":
    main()
