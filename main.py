


import ffmpeg


input_file_name='Batch_Night_20_11.mp4' # LR
input_file_name='Batch_Night_20_11_HD.mp4' # HR
base='Night_20_11'

# 20:11:53        20:11:55
process = (ffmpeg
        .input(input_file_name,ss=('00:00:48'),to=('00:00:50') ) # 20:11:05
 .filter('fps', fps=10, round = 'up')
 .output("%s-%%04d.jpg"%(base), **{'qscale:v': 3})
 .run_async(pipe_stdin=True))


# HR +48 .input(input_file_name,ss=('00:00:48'),to=('00:00:50') ) # 20:11:05
# LR +41 .input(input_file_name,ss=('00:00:41'),to=('00:00:43') )
