# simple_distributed_machine_learning

requirement: 
+ python
+ numpy
+ pytorch: CPU or CUDA 10.2

to running code:
   
  |****************|     feed-forward      |****************|
  |     server1    |  ---->---->---->----  |     server2    |
  |     rank=0     |                       |     rank=1     |
  |****************|                       |****************|
          |                                         |
          |             back-propagation            |
          |--------<--------<--------<--------<-----|


/usr/bin/python3 distributed.py --rank=0 --world_size=2 --interface=eth0 --master_addr=192.168.10.128 --master_port=2308
