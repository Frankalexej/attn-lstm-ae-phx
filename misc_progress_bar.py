import sys

def draw_progress_bar(iteration, total, bar_length=50, content=""):
    """
    Draw a text-based progress bar in the console.

    Parameters:
    - iteration (int): The current iteration.
    - total (int): The total number of iterations.
    - bar_length (int): The length of the progress bar.

    Example usage:
    for i in range(0, total + 1):
        draw_progress_bar(i, total)
    """
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\r[{arrow + spaces}] {int(progress * 100)}%\t{content}')
    sys.stdout.flush()



if __name__ == "__main__": 
    import time

    total_iterations = 100

    for i in range(0, total_iterations + 1):
        draw_progress_bar(i, total_iterations, content=str(i))
        time.sleep(0.1)  # Simulate some work being done
