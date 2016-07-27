import matplotlib.pyplot as plt

fp = open('trend.log')
lines = ''.join( fp.readlines() ).split('\n')

i_end = len(lines)
for i_pre in range(len(lines)):
    i = len(lines) - i_pre - 1
    if lines[i] == 'last':
        i_stt = i + 2
        break

plt.plot( [float(x) for x in lines[i_stt:i_end - 1]] )
plt.ylabel('validation error at each epoch')
plt.axis([0, i_end-i_stt, 0, 100])
plt.show()

