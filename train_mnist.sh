#!/bin/bash
export LD_LIBRARY_PATH=/vol/vssp/signsrc/externalLibs/cudnn-8.0-linux-x64-v6.0/lib64:$LD_LIBRARY_PATH                                                                                                                                                                          
exec python mnist_deep.py --num_iterations 10
