import ppgs
import torch
import numpy as np
import matplotlib.pyplot as plt
print(ppgs.REPRESENTATION)
audio_file = "sendo.wav"
ppg = ppgs.from_file(audio_file,checkpoint="runs/ppgs/00200000.pt",gpu=0).cpu().to(torch.float32).numpy()
print(ppg.shape)
labels= [
	'a',
	'i',
	'u',
	'e',
	'o',
	'N',
	'w',
	'y',
	'j',
	'my',
	'ky',
	'dy',
	'gy',
	'ny',
	'hy',
	'ry',
	'py',
	'p',
	't',
	'k',
	'ts',
	'ch',
	'b',
	'd',
	'g',
	'z',
	'm',
	'n',
	's',
	'sh',
	'h',
	'f',
	'r',
	'q',
	'<silent>']
mas = []
for i in range(ppg.shape[1]):
    print(labels[np.argmax(ppg.T[i])],end=',')


plt.figure(figsize=(20,8))
plt.yticks(ticks=np.arange(len(labels)), labels=labels)

plt.imshow(ppg, cmap='viridis', aspect='auto')
plt.colorbar() 
plt.title('PPGs Visualization')
plt.ylabel('phonemes')
plt.xlabel('frames')
plt.savefig("ppg_sendo")