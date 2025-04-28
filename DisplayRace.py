import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt

from Courses import create_courses
from NeuralNet import NeuralNetwork
from NeuralNetConfigs import BLOCKER_NN_CONFIG
from TrainPodBlocker import PodBlockerGeneticAlgorithm
from TrainPodRacer import PodRacerGeneticAlgorithm, RACER_NN_CONFIG

X_MAX = 16000
Y_MAX = 9000
POD_SIZE = 400
CHECKPOINT_SIZE = 300
TIME_PER_FRAME = 0.05


def generate_and_display_race(racer_config, blocker_config):
    course = create_courses(1)[0]
    racer = NeuralNetwork.create_from_json(racer_config, RACER_NN_CONFIG)
    if blocker_config is not None:
        blocker = NeuralNetwork.create_from_json(blocker_config, BLOCKER_NN_CONFIG)
        score, paths, next_checkpoints, inputs = PodBlockerGeneticAlgorithm.evaluate_racer_and_blocker(course, racer, blocker, True)
        plot_pod_race(course.checkpoints, paths, None, None)
    else:
        score, path, next_checkpoints, inputs = PodRacerGeneticAlgorithm.evaluate_racer(course, racer, True)
        plot_pod_race(course.checkpoints, [path], next_checkpoints, inputs)


def plot_pod_race(checkpoints, paths, next_checkpoints, inputs):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes(xlim=(0, X_MAX), ylim=(0, Y_MAX))
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()

    checkpoint_icons = []
    for checkpoint in checkpoints:
        circle = plt.Circle((checkpoint[0], checkpoint[1]), CHECKPOINT_SIZE, color='r')
        checkpoint_icons.append(circle)

    number_of_pods = len(paths)
    pod_icons = []
    for pod_id in range(number_of_pods):
        pod = plt.Circle((paths[pod_id][0][0], paths[pod_id][0][1]), POD_SIZE, color='b' if pod_id == 0 else 'g')
        pod_icons.append(pod)

    output_template = 'round %i  steer %i  thrust %s  next checkpoint %i'
    output_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        for j in range(len(checkpoints)):
            checkpoint_icons[j].center = (checkpoints[j][0], checkpoints[j][1])
            ax.add_patch(checkpoint_icons[j])
        for _pod_id in range(number_of_pods):
            _pod = pod_icons[_pod_id]
            _pod.center = (paths[_pod_id][0][0], paths[_pod_id][0][1])
            ax.add_patch(_pod)
        return *checkpoint_icons, *pod_icons, output_text

    def animate(i):
        for _pod_id in range(number_of_pods):
            pod_icons[_pod_id].center = (paths[_pod_id][i][0], paths[_pod_id][i][1])
        if inputs is not None:
            angle, thrust = inputs[min(i, len(inputs) - 1)]
            output_text.set_text(output_template % (i, angle, str(thrust), next_checkpoints[i]))
        return *checkpoint_icons, *pod_icons, output_text

    _ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(paths[0]), interval=TIME_PER_FRAME * 1000, blit=True)
    plt.show()


def plot_pod_paths(checkpoints, paths, pause_time):
    plt.figure(figsize=(5, 4))
    ax = plt.axes(xlim=(0, X_MAX), ylim=(0, Y_MAX))
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()

    checkpoint_icons = []
    for checkpoint in checkpoints:
        circle = plt.Circle((checkpoint[0], checkpoint[1]), CHECKPOINT_SIZE, color='r')
        checkpoint_icons.append(circle)
        ax.add_patch(circle)
    start_position = paths[0]['path'][0]
    pod = plt.Circle((start_position[0], start_position[1]), POD_SIZE, color='b')
    ax.add_patch(pod)
    for path in paths:
        string_path = mpath.Path(path['path'])
        track = mpatches.PathPatch(string_path, facecolor="none", lw=2)
        ax.add_patch(track)

    plt.show(block=False)
    plt.pause(pause_time)
    plt.close()


if __name__ == '__main__':
    generate_and_display_race(open('nn_data/live_racer_nn_config.txt').readlines()[0],
                              open('nn_data/live_blocker_nn_config.txt').readlines()[0])
