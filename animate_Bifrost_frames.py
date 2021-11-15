import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


fig = plt.figure()

im = plt.imshow(bp[0, 0, :, :, 0], animated=True, vmin=-10, vmax=10,
                cmap="bwr", origin="lower")
plt.colorbar(im, label="Velocity")

def updatefig(i):
    im.set_array(bp[i, 0, :, :, 0])
    plt.title(f"Time = {i*10} seconds")

    return im,


anim = animation.FuncAnimation(fig, updatefig, interval=10,
                               frames=149, blit=True, repeat=True)
f = "bp_velocity.mp4"
writervideo = animation.FFMpegWriter(fps=20)
anim.save(f, writer=writervideo)
plt.show()
