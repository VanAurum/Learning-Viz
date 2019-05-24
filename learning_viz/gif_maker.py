

#3rd party imports
import imageio


def make_gif(files, directory, gif_name='default', fps=5):
    '''A utility function to make a gif from a list of image files.

    Parameters:
    -----------
    files : list 
        A list of file names that are images.
    directory : string 
        The directory to save the final GIF 
    gif_name : string 
        The name to give the gif (without .gif file extension)
    fps : int, 
        The frames per second the gif should play at.        
    '''
    images=[]
    for file in files:
        images.append(imageio.imread(file))
    imageio.mimsave(directory+gif_name+'.gif', images, fps=fps)