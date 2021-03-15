from pathlib import Path
import subprocess, os, sys
import traceback
import time
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import re

# find a folder by pattern
import glob

# 1. Remove try catch for program run
# 2. Add dataset choosing
# 3. Change dataset to big

"""
48 experiments, they will take approx ~ 15 hours.
"""

def parse_blocksize(folder_path, file_name):
    """
    Function parses blktrace logs to extract blocksize distribution data
    """
    # (echo "IO_Size"; blkparse optane -a read | awk '/ Q / { if ($10 != 0) { print $10/2 } else print 0 }') >optane_read_sizes.csv

    read_size = "(echo 'IO_Size'; blkparse " + file_name + " -a read -D " + folder_path + " | awk '/ C / { if ($10 > 1) { print $10/2 } else print 0 }') >" + folder_path + file_name + "_read_sizes.csv"

    subprocess.call(read_size, shell=True)

    write_size = "(echo 'IO_Size'; blkparse " + file_name + " -a write -D " + folder_path + " | awk '/ C / { if ($10 > 1) { print $10/2 } else print 0 }') >" + folder_path + file_name + "_write_sizes.csv"

    subprocess.call(write_size, shell=True)

def blkparse(folder_path, file_name):
    """
    Function parses blktrace logs to get binary file and produces blkparse output of avg thgroughput, read size and write size
    """

    blkparse = "blkparse -i " + folder_path + file_name + ".blktrace. -d " + file_name + ".bin > " + folder_path + file_name +  "_blkparse.txt"

    subprocess.run(blkparse, shell=True, capture_output=True, text=True)

    with open(os.path.join(folder_path, file_name +  "_blkparse.txt"), 'r') as input_file:
        last_io_line = None
        throughput = None
        for line in input_file:
            #print(line)
            match_io_size = re.search(r"Reads", line)
            if match_io_size:
                last_io_line = line
            
            match_throughput = re.search(r"Throughput", line)
            if match_throughput:
                throughput = line

        throughput = "Avg " + throughput
        #print(throughput)
        required_size = last_io_line.split()
        read_size = required_size[3]
        write_size = required_size[7]
        #print(read_size)
        #print(write_size)

        #os.remove(os.path.join(folder_path, "output.txt"))

    return {'read_size': read_size , 'write_size': write_size, 'throughput': throughput}

def get_mean_mode_blocksize(input_file):
    """
    Function return mean and mode of blocksize of specified input file
    """

    df = pd.read_csv(input_file, delim_whitespace=True)

    try: 
        mode = df["IO_Size"].mode().values[0]
        mean = round(df["IO_Size"].mean(), 2)
    except IndexError:
        print("IndexError")
        mode = "IndexError"
        mean = "IndexError"

    #print("Mode: " + str(mode))
    #print("Mean: " + str(mean))

    return {'mean': mean, 'mode': mode}

def get_mean_mode_blocksize_read_write(folder_path, file_name):
    """
    Function produces mean and mode of read and write log files
    """
    read_file = folder_path + file_name + "_read_sizes.csv"
    write_file = folder_path + file_name + "_write_sizes.csv"
    
    return({"read": get_mean_mode_blocksize(read_file), "write": get_mean_mode_blocksize(write_file)})

def btt(folder_path, file_name):
    """
    Generating latency, throughput, IOPS and block numbers logs files
    """

    btt = "btt -i " + file_name +".bin -l " + file_name + "_latency -o " + file_name + ".out -B " + file_name + ".dump_blocknos"
    subprocess.call(btt, shell=True, stdout=subprocess.DEVNULL)

    subprocess.run(["mv " + file_name + ".bin " + folder_path], shell=True)

    latency_file = glob.glob(file_name + "_latency*")
    try: 
        #print("Found latency file: " + latency_file[0])
        subprocess.run(["mv " + latency_file[0] + " " + folder_path], shell=True)
    except:
        print("Latency file not found")

    throughput_file = glob.glob("sys_mbps_fp.dat")
    try: 
        #print("Found throughput file: " + throughput_file[0])
        new_file_name = file_name + "_" + throughput_file[0]
        subprocess.run(["mv " + throughput_file[0] + " " + new_file_name], shell=True)
        subprocess.run(["mv " + new_file_name + " " + folder_path], shell=True)
    except:
        print("Throughput file not found")

    iops_file = glob.glob("sys_iops_fp.dat")
    try: 
        #print("Found iops file: " + iops_file[0])
        new_file_name = file_name + "_" + iops_file[0]
        subprocess.run(["mv " + iops_file[0] + " " + new_file_name], shell=True)
        subprocess.run(["mv " + new_file_name + " " + folder_path], shell=True)
    except:
        print("IOPS file not found")

def plot_latency(folder_path, file_name):
    latency_file = glob.glob(folder_path + file_name + "_latency_*")
    #print("Folder path: " + folder_path)
    #print(latency_file)

    data = pd.read_csv(latency_file[0], header=None, usecols=[0,1], delimiter=' ')

    time_plot = data[0]
    latency = data[1]

    plt.plot(time_plot, latency, color='blue',
            linestyle='-')

    #plt.legend()
    plt.suptitle('Latency ' + file_name)
    plt.title('thread: ' + str(th) +  ' blocksize: ' + str(bs) + ' buffer size: ' + str(buf), fontsize=10)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Latency')
    plt.grid()

    plt.savefig(folder_path +  file_name + '_latency.png')

    plt.cla()
    plt.clf()

def plot_throughput(folder_path, file_name):
    throughput_file = os.path.join(folder_path, file_name + "_sys_mbps_fp.dat")
    #print("Folder path: " + folder_path)
    #print(latency_file)

    data = pd.read_csv(throughput_file, header=None, usecols=[0,1], delimiter=' ')

    time_plot = data[0]
    throughput = data[1]

    plt.plot(time_plot, throughput, color='red',
            linestyle='-')

    #plt.legend()

    plt.suptitle('Throughput ' + file_name)
    plt.title('thread: ' + str(th) +  ' blocksize: ' + str(bs) + ' buffer size: ' + str(buf), fontsize=10)
    
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Throughput (MiB/s)')
    plt.grid()
    
    #plt.show()

    plt.savefig(folder_path + file_name + '_throughput.png')

    plt.cla()
    plt.clf()

def plot_latency_throughput(folder_path, file_name):
    latency_file = glob.glob(folder_path + file_name + "_latency_*")
    throughput_file = os.path.join(folder_path, file_name + "_sys_mbps_fp.dat")

    latency_data = pd.read_csv(latency_file[0], header=None, usecols=[0,1], delimiter=' ')
    throughput_data = pd.read_csv(throughput_file, header=None, usecols=[0,1], delimiter=' ')

    
    time_lat = latency_data[0]
    latency = latency_data[1]
    time_thr = throughput_data[0]
    throughput = throughput_data[1]
    
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    
    ax.plot(time_thr, throughput, color='red',
            linestyle='-', label='Throughput')
    
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel('Throughput (MiB/s)')
    
    
    ax2.plot(time_lat, latency, color='blue',
            linestyle='-', label='Latency')
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Latency (µs)')
    ax2.yaxis.set_label_position('right')
    ax2.xaxis.set_visible(False)
    

    ax.legend(loc=(0.07, 0.85))
    ax2.legend(loc=(0.07, 0.75))


    plt.suptitle('Throughput and Latency ' + file_name)
    plt.title('thread: ' + str(th) +  ' blocksize: ' + str(bs) + ' buffer size: ' + str(buf), fontsize=10)
    
    
    #plt.xlabel('Time (seconds)')
    #plt.ylabel('Throughput (MiB/s)')
    plt.grid()
    
    #plt.show()
    plt.gcf().subplots_adjust(right=0.85)

    plt.savefig(folder_path + file_name + '_throughput_latency.png')

    plt.cla()
    plt.clf()

def plot_blocksize(input_file, folder_path, file_name, workload_type):
    """
    Function to plot blocksize data a as barchart of blocksize distribution
    """
    plt.rcParams['figure.figsize'] = [9, 5]

    df=pd.read_csv(input_file, delim_whitespace=True)
    
    total_ios = len(df.index)
    min_size = 0
    max_size = df["IO_Size"].max()
    print(max_size)
    major_locator = 8
    minor_locator = 4
    plot_bin = 4
    # rearranging bins with big distribution of IO size
    if max_size >= 1024:
        major_locator = 64
        minor_locator = 32
        plot_bin = 32
    elif max_size >= 4096:
        major_locator = 256
        minor_locator = 128
        plot_bin = 256

    
    mode = df["IO_Size"].mode()
    mean = df["IO_Size"].mean()
    counts = df["IO_Size"].value_counts()
    #print(counts)
    
    ax_list = df.plot(kind='hist',
                    subplots=True,
                    sharex=True,
                    sharey=True,
                    title='IO blocksizes ' + file_name + ' ' + workload_type + ' th: ' + str(th) +  ' bs: ' + str(bs) + ' buf size: ' + str(buf),
                    color='g',
                    bins=np.arange(min_size - 2, max_size + plot_bin, plot_bin)) # produces problems. Было 4
    ax_list[0].set_xlim(0, max_size)
    ax_list[0].set_xlabel('IO size (KiB)')
    ax_list[0].tick_params(axis='x', rotation=-45)
    # Major ticks every 4KiB, minor every 2 KiB in between. Controls signs
    ax_list[0].xaxis.set_major_locator(plt.MultipleLocator(major_locator)) # was 8
    ax_list[0].xaxis.set_minor_locator(plt.MultipleLocator(minor_locator)) # was 4
    # Convert y tick labels to percentages by multiplying them by
    # 100.
    y_vals = ax_list[0].get_yticks()

    #print(y_vals)
    ax_list[0].set_yticklabels(['{:,.1%}'.format(v/total_ios) for v in y_vals])
    
    ax_list[0].grid(b=True, linestyle=':', color='#666666')
    ax_list[0].legend([workload_type+ " IOs"])
    
    # x axis label doesn't seem to fit on unless the bottom is
    # extended a little.
    plt.subplots_adjust(bottom=0.15)
    
    # Add an info box with mean/mode values in.
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax_list[0].text(0.2,
                    0.95,
                    "Mode: %3uKiB\nMean: %6.2fKiB" % (mode[0], mean),
                    transform=ax_list[0].transAxes,
                    fontsize=14,
                    verticalalignment='top',
                    bbox=props)

    plt.savefig(folder_path + file_name + '_blocksize_' + workload_type.lower() + '.png')

    plt.cla()
    plt.clf()

def plot_blocksize_read_write(folder_path, file_name):
    """
    Function, which plots blocksize distribution for writes and reads
    """
    read_input = folder_path + file_name + "_read_sizes.csv"
    write_input = folder_path + file_name + "_write_sizes.csv"

    plot_blocksize(read_input, folder_path, file_name, "Read")
    plot_blocksize(write_input, folder_path, file_name, "Write")

def plot_throughput_and_accumulated_read_write(folder_path, file_name):
    """
    Parses throughput data and plots accumulated read write size plot.
    """

    def get_throughput_data(folder_path, file_name):
        """
        Uses blkparse to parse blktrace output logs to get throughput data.
        Used for in function for creating read write throughput plots and for creating
        accumulated read write plots.
        """

        # read logs  
        blkparse_read_throughput = "blkparse -i " + folder_path + file_name + ".blktrace. -f '%5T %a %N\n' -a read -o " + folder_path + file_name + "_throughput_raw_read.csv"
        blkparse_read_throughput_filtered = "(echo 'seconds operation io_size'; grep ' C ' " + folder_path + file_name + "_throughput_raw_read.csv) > " + folder_path + file_name +  "_throughput_filtered_read.csv"

        subprocess.call(blkparse_read_throughput, shell=True, stdout=subprocess.DEVNULL)
        subprocess.call(blkparse_read_throughput_filtered, shell=True)

        # write logs
        blkparse_write_throughput = "blkparse -i " + folder_path + file_name + ".blktrace. -f '%5T %a %N\n' -a write -o " + folder_path + file_name + "_throughput_raw_write.csv"
        blkparse_write_throughput_filtered = "(echo 'seconds operation io_size'; grep ' C ' " + folder_path + file_name + "_throughput_raw_write.csv) > " + folder_path + file_name +  "_throughput_filtered_write.csv"

        subprocess.call(blkparse_write_throughput, shell=True, stdout=subprocess.DEVNULL)
        subprocess.call(blkparse_write_throughput_filtered, shell=True)

    def generate_accumulated_plot(folder_path, file_name, workload_type):
        """
        Generates accumulated IO plot.
        """
        data = pd.read_csv(folder_path + file_name, delimiter=';')
        #print(data)

        time_plot = data['seconds']
        throughput = data['io_size']

        color = 'red'
        if workload_type == 'write':
            color = 'blue'

        plt.plot(time_plot, throughput, color=color,
                linestyle='-')

        min_size = 0
        max_size = time_plot.max()

        # plt.xticks(np.arange(min_size, max_size, 10))

        #plt.legend()


        plt.suptitle('Accumulated IO size ' + workload_type + ' ' + file_name)
        plt.title('thread: ' + str(th) +  ' blocksize: ' + str(bs) + ' buffer size: ' + str(buf), fontsize=10)
        
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('IO size (MiB)')
        plt.grid()
        
        #plt.show()

        plt.savefig(folder_path + file_name[:-4] + '.png')

        plt.cla()
        plt.clf()
    
    def generate_throughput_read_write_plot(folder_path, file_name, workload_type):

        data = pd.read_csv(folder_path + file_name, delimiter=';')
        #print(data)

        color = 'red'
        if workload_type == 'write':
            color = 'blue'

        time_plot = data['seconds']
        throughput = data['io_size']

        plt.plot(time_plot, throughput, color=color,
                linestyle='-')

        min_size = 0
        max_size = time_plot.max()

        # plt.xticks(np.arange(min_size, max_size, 10))

        #plt.legend()

        plt.suptitle('Throughput ' + workload_type + ' ' + file_name)
        plt.title('thread: ' + str(th) +  ' blocksize: ' + str(bs) + ' buffer size: ' + str(buf), fontsize=10)
        
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Throughput (MiB/s)')
        plt.grid()
        
        #plt.show()

        plt.savefig(folder_path + file_name[:-4] + '.png')

        plt.cla()
        plt.clf()

    get_throughput_data(folder_path, file_name)
    
    ### preparing throughput data for visualization
    # writes
    data_write = pd.read_csv(folder_path + file_name + "_throughput_filtered_write.csv", delim_whitespace=True)

    # sum by group and translate values to MiBs
    grouped_write_data = data_write.groupby(pd.Grouper(key="seconds")).sum()
    grouped_write_data.io_size = grouped_write_data.io_size / 1024 / 1024

    # save dataframe as csv
    grouped_write_data.to_csv(folder_path + file_name + "_throughput_write.csv", columns=['io_size'], sep=';')

    # reads
    data_read = pd.read_csv(folder_path + file_name + "_throughput_filtered_read.csv", delim_whitespace=True)

    # sum by group and translate values to MiBs
    grouped_read_data = data_read.groupby(pd.Grouper(key="seconds")).sum()
    grouped_read_data.io_size = grouped_read_data.io_size / 1024 / 1024

    # save dataframe as csv
    grouped_read_data.to_csv(folder_path + file_name + "_throughput_read.csv", columns=['io_size'], sep=';')

    generate_throughput_read_write_plot(folder_path, file_name + "_throughput_read.csv", "read")
    generate_throughput_read_write_plot(folder_path, file_name + "_throughput_write.csv", "write")

    ### Caluclating cumsum for reads and writes
    # reads
    cumsum_read = grouped_read_data.io_size.cumsum()
    #print(cumsum)
    output_file_name_read = file_name + '_cumsum_read.csv'

    cumsum_read.to_csv(folder_path + output_file_name_read, header=['io_size'], columns=['io_size'], sep=';')
    generate_accumulated_plot(folder_path, output_file_name_read, 'read')

    # writes
    cumsum_write = grouped_write_data.io_size.cumsum()
    #print(cumsum)
    output_file_name_write = file_name + '_cumsum_write.csv'

    cumsum_write.to_csv(folder_path + output_file_name_write, header=['io_size'], columns=['io_size'], sep=';')

    generate_accumulated_plot(folder_path, output_file_name_write, 'write')

############################################ Main started

started_time = datetime.now().strftime('%H:%M:%S %d-%m-%Y')
#print(current_time)

# input values
folder_with_experiments = 'matrix_test'
file_name = "optane_mt_mean"


threads = [0, 3, 7, 11]
buffer_size = [128, 2048, 8192]
block_size = [1, 100, 1000, 8000, 999]

file_pointer = 1
app_path = None

# block size 999 - bigget possbile size of block size

# threads = [0, 3, 7, 11]
# buffer_size = [8192, 1024, 128]
# block_size = [1, 10, 100, 999]

total_tests = len(threads) * len(buffer_size) * len(block_size)

# experiment iterator. Used for creating folders
test_number_iterator = 1

start = time.time()
print('VVV' + '\n')

app_output = {}
mean_mode_output = {}
blkparse_read_write_sizes = {}

"""
Output dict structure:
{test_number: {
    app_output: [line1, line2, line3, ...] - str: list
    mean: 'mean' - str: str
    mode: 'mode' - str: str
    read_size: 'read_size' - str: str
    write_size: 'write_size' - str: str
    throughput: 'throughput' - str: str
}}
"""
file_output = {}
csv_output = []

for th in threads:
    for buf in buffer_size:
        for bs in block_size:
            working_folder = os.getcwd()
            folder_path = working_folder + '/' + folder_with_experiments + "/test" + str(test_number_iterator) + "_" + str(th) + "th_" + str(buf) + "buf_" + str(bs) + "bs/"
            #folder_path = "./blktrace_test/test" + str(test_number_iterator) + "_" + str(th) + "th_" + str(buf) + "buf_" + str(bs) + "bs/"
            Path(folder_path).mkdir(parents=True, exist_ok=True)

            now = time.time()
            progress = 100 / total_tests * test_number_iterator         

            progress_output = "Completed: " + str(round(progress, 0)) + "% " + " | Elapsed time: " + str(round(now - start, 1)) + " seconds" + " | === TEST " + str(test_number_iterator) + "/" + str(total_tests) + " === threads: " + str(th) + " | buffer_size: " + str(buf) + " | block_size: " + str(bs) + "|||\n"

            sys.stdout.write("\r%s" % progress_output)
            sys.stdout.flush()

            


            ############# Replacement buffer size 

            replacement_buf = 'sed -i "s/int max_mbytes = [0-9]*/int max_mbytes = ' + str(buf) + '/g" /media/optane/cpp_playground/mt_mean_simple.cpp'

            subprocess.call(replacement_buf, shell=True)
            #print("Buffer size changed!")

            if bs == 999:
                replacement_bs = 'sed -i "s@[0-9]*, [0-9]*, // nXSize, nYSize@cols, nlines, // nXSize, nYSize@g" /media/optane/cpp_playground/mt_mean_simple.cpp'

            elif bs == 1:
                replacement_bs = 'sed -i "s@cols, nlines, // nXSize, nYSize@' + str(bs) + ', ' + str(bs) + ', // nXSize, nYSize@g" /media/optane/cpp_playground/mt_mean_simple.cpp'

            else:
                replacement_bs = 'sed -i "s@[0-9]*, [0-9]*, // nXSize, nYSize@' + str(bs) + ', ' + str(bs) + ', // nXSize, nYSize@g" /media/optane/cpp_playground/mt_mean_simple.cpp'

            subprocess.call(replacement_bs, shell=True)
            #print("Blocksize changed!")


            ############# Compiling

            compiling = "/usr/bin/g++ -g /media/optane/cpp_playground/mt_mean_simple.cpp -I/usr/include/gdal -lgdal -fopenmp -larmadillo -O3 -o /media/optane/cpp_playground/build/mt_mean_simple"
            subprocess.call(compiling, shell=True)
            #print("Program compiled!")

            ############# Choosing dataset
            if file_pointer == 1:
                app_path = '/media/optane/cpp_playground/mt_mean_simple_execute1.sh'
                file_pointer = file_pointer + 1

            elif file_pointer == 2:
                app_path = '/media/optane/cpp_playground/mt_mean_simple_execute2.sh'
                file_pointer = file_pointer + 1

            elif file_pointer == 3:
                app_path = '/media/optane/cpp_playground/mt_mean_simple_execute3.sh'
                file_pointer = 1

            # sync the memory and disk
            subprocess.run(["sudo", "sync"])

            # clearing the OS caches
            subprocess.run(["sudo", "sh", "-c", 'echo 1 >/proc/sys/vm/drop_caches'])
            subprocess.run(["sudo", "sh", "-c", 'echo 2 >/proc/sys/vm/drop_caches'])
            subprocess.run(["sudo", "sh", "-c", 'echo 3 >/proc/sys/vm/drop_caches'])
            time.sleep(5)

            ############# blktrace run

            # creates new thread for blktrace
            blktrace = subprocess.Popen(['sudo', 'blktrace', '-d', '/dev/nvme1n1', '-o', 'optane', '-D', folder_path], preexec_fn=os.setpgrp, stdout=subprocess.DEVNULL) # something long running
            #print("blktrace spawned with PID: %s" % p.pid)


            ############# Application run
            app = subprocess.Popen(["numactl", "--physcpubind=+0-" + str(th), app_path], stdout=subprocess.PIPE)
            #print("App spawned with PID: %s" % app.pid)

            
            '''
            Collecting I/O logs from proc
            '''
            while app.poll() == None:
                time.sleep(2)
                proc = "cat /proc/" + str(app.pid) + "/io"
                proc_run = subprocess.run(proc, shell=True, capture_output=True, text=True)
                txt = proc_run.stdout
                with open(os.path.join(folder_path, "proc.txt"), 'w') as input_file:
                    input_file.write(txt)
                #print(txt)

            app_output = app.communicate()

            #print(output)
            rchar = None
            wchar = None    
            read_bytes = None
            write_bytes = None

            with open(os.path.join(folder_path, file_name + "_proc.txt"), 'r') as input_file:
                i = 0
                for line in input_file:
                    if (i == 0):
                        rchar = line
                    if (i == 1):
                        wchar = line
                    if (i == 4):
                        read_bytes = line
                    if (i == 5):
                        write_bytes = line
                    i += 1

            rchar = rchar.split(" ")
            rchar = rchar[1]
            rchar = int(rchar) / 1024 / 1024

            wchar = wchar.split(" ")
            wchar = wchar[1]
            wchar = int(wchar) / 1024 / 1024

            read_bytes = read_bytes.split(" ")
            read_bytes = read_bytes[1]
            read_bytes = int(read_bytes) / 1024 / 1024

            write_bytes = write_bytes.split(" ")
            write_bytes = write_bytes[1]
            write_bytes = int(write_bytes) / 1024 / 1024

            #print("Reading:" + str(proc_read_bytes) + " MiB, Writing: " + str(proc_write_bytes) + " MiB")

            # app_output[test_number_iterator] = ["Test " + str(test_number_iterator) + ". " + "Thread: " + str(th) + " Blocksize: " + str(bs) + " Buffer size: " + str(buf) + "\n"]
            # app_output[test_number_iterator].append(app.stdout + "\n")
            file_output[test_number_iterator] = {}
            file_output[test_number_iterator]['app_output'] = ["Test " + str(test_number_iterator) + ". " + "Thread: " + str(th) + " Blocksize: " + str(bs) + " Buffer size: " + str(buf) + "\n"]
            file_output[test_number_iterator]['app_output'].append(str(app_output[0].decode()) + "\n")
            file_output[test_number_iterator]['proc_rchar'] = "Proc rchar: " + str(rchar) + " MiB"
            file_output[test_number_iterator]['proc_wchar'] = "Proc wchar: " + str(wchar) + " MiB"
            file_output[test_number_iterator]['proc_read_bytes'] = "Proc read_bytes: " + str(read_bytes) + " MiB"
            file_output[test_number_iterator]['proc_write_bytes'] = "Proc write_bytes: " + str(write_bytes) + " MiB"
            #file_output[test_number_iterator]['app_output'].append(app.stdout + "\n")

            # addding folder with output file
            file_output[test_number_iterator]['folder_path'] = "Folder path: " + folder_path

            #collecting data for csv
            app_output = app_output[0].decode()
            app_output = app_output.split(' ')
            #print('Elapsed time: ' + app_output[24])
            try:
                csv_output.append({'elapsed_time': app_output[24], 'read_time': app_output[15], 'thread': th, 'blocksize': bs, 'buffer_size': buf, 'proc rchar, MiB': str(rchar), 'proc read_bytes, MiB': str(read_bytes), 'blktrace read_size, MiB': None, 'folder': folder_path})
            except Exception:
                csv_output.append({'elapsed_time': traceback.format_exc(), 'read_time': traceback.format_exc(), 'thread': th, 'blocksize': bs, 'buffer_size': buf, 'proc rchar, MiB': None, 'proc read_bytes, MiB': None, 'blktrace read_size, MiB': None, 'folder': folder_path})


            time.sleep(60)
            blktrace.terminate()
            
            #print("blktrace terminated")

            ######################## Parsing blocksize
            parse_blocksize(folder_path, file_name)
            mean_mode = get_mean_mode_blocksize_read_write(folder_path, file_name)

            #print("Mean: " + str(mean_mode['mean']) + " KiB")
            #print("Mode: " + str(mean_mode['mode']) + " KiB")

            #mean_mode_output[test_number_iterator] = ["Mean: " + str(mean_mode['mean']) + " KiB, ", "Mode: " + str(mean_mode['mode']) + " KiB"]

            file_output[test_number_iterator]['mean'] = "Mean: " + str(mean_mode['mean']) + " KiB"
            file_output[test_number_iterator]['mode'] = "Mode: " + str(mean_mode['mode']) + " KiB"

            ######################## Parsing latency

            blkparse(folder_path, file_name)
            blk_parse_output = blkparse(folder_path, file_name)
            blkparse_read_write_sizes[test_number_iterator] = ["Read size: " + blk_parse_output['read_size'], "Write size: " + blk_parse_output['write_size']]

            file_output[test_number_iterator]['read_size'] = "Blktrace read size: " + blk_parse_output['read_size']
            file_output[test_number_iterator]['write_size'] = "Blktrace write size: " + blk_parse_output['write_size']
            file_output[test_number_iterator]['throughput'] = blk_parse_output['throughput']
            #print("Blkparse read write sizes " + str(blkparse_read_write_sizes))

            csv_output[test_number_iterator-1]['blktrace read_size, MiB'] = blk_parse_output['read_size'][:-3]

            # change folder?? os.child
            # generating log output with btt
            btt(folder_path, file_name)

            ######################## Generating plots

            plot_latency(folder_path, file_name)
            plot_throughput(folder_path, file_name)
            plot_latency_throughput(folder_path, file_name)
            plot_blocksize_read_write(folder_path, file_name)
            plot_throughput_and_accumulated_read_write(folder_path, file_name)

            ######################## Test number increase
            test_number_iterator = test_number_iterator + 1


end = time.time()
finished_time = datetime.now().strftime('%H:%M:%S %d-%m-%Y')

# save output to csv
index_list = list(range(1, total_tests+1))
#print(index_list)
final_csv_df = pd.DataFrame(csv_output, index=index_list)
#print(final_csv_df)
final_csv_df = final_csv_df.sort_values('elapsed_time')
#print(final_csv_df)
final_csv_df.to_csv(file_name + '.csv', sep=';', encoding='utf-8')

# create a report
with open(file_name + ".txt", "a") as myfile:
    myfile.write("==========="+ "\n")
    myfile.write("Experiments started: " + str(started_time) + "\n")
    myfile.write("Experiments finished: " + str(finished_time) + "\n")
    myfile.write("Total time elapsed: " + str(round(end - start, 1)) + " seconds" + " (" + str(round(((end - start)/60/60), 1)) + " hours)" + "\n")
    myfile.write("Total tests: " + str(total_tests) + "\n")
    myfile.write("Experiment parameters: " + "\n")
    myfile.write("Threads: " + str(threads) + "\n")
    myfile.write("Block sizes: " + str(block_size) + "\n")
    myfile.write("Buffer sizes " + str(buffer_size) + "\n")
    myfile.write("\n")
    myfile.write("\n")

# dict structure: { experiment_number: {parameter: value}}
# iterating over values in dict and writing them to the file
    for value in file_output.values():
        for key, val in value.items():
            if (key) == 'app_output':
                for v in val:
                    myfile.write(v)
            else:
                myfile.write(val)
                myfile.write("\n")


        myfile.write("-----------"+ "\n")
        myfile.write("\n")


print('\n')
print("Total tests: " + str(total_tests))
print("Total time elapsed: " + str(round(end - start, 1)) + " seconds" + " ("+ str(round(((end - start)/60/60), 1)) + " hours)")
print('\n')
