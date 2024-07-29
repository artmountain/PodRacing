import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.animation as animation

X_MAX = 16000
Y_MAX = 9000
POD_SIZE = 500
CHECKPOINT_SIZE = 300
TIME_PER_FRAME = 0.05
# TIME_PER_FRAME = 1 # todo


def plot_pod_race(checkpoints, path, inputs, nn_data):
    print(len(path), len(inputs))
    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes(xlim=(0, X_MAX), ylim=(0, Y_MAX))
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()

    checkpoint_icons = []
    for checkpoint in checkpoints:
        circle = plt.Circle((checkpoint[0], checkpoint[1]), CHECKPOINT_SIZE, color='r')
        checkpoint_icons.append(circle)
    pod = plt.Circle((path[0][0], path[0][1]), POD_SIZE, color='b')
    ax.add_patch(pod)

    output_template = 'round %i  steer %i'
    output_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        for j in range(3):
            checkpoint_icons[j].center = (checkpoints[j][0], checkpoints[j][1])
            ax.add_patch(checkpoint_icons[j])
        pod.center = (path[0][0], path[0][1])
        ax.add_patch(pod)
        return checkpoint_icons[0], checkpoint_icons[1], checkpoint_icons[2], pod, output_text

    def animate(i):
        pod.center = (path[i][0], path[i][1])
        angle = int(inputs[min(i, len(inputs) - 1)][0])
        output_text.set_text(output_template % (i, angle))
        print([[round(x, 2) for x in nn_data[i][j]] for j in range(2)])
        return checkpoint_icons[0], checkpoint_icons[1], checkpoint_icons[2], pod, output_text

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(path), interval=TIME_PER_FRAME * 1000, blit=True)
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
