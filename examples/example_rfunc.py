from pastas.rfunc import *
import matplotlib.pyplot as plt
r=Gamma()
p=r.set_parameters('test')
plt.plot(r.step(p['initial']), label='Gamma')

r=Bruggeman()
p=r.set_parameters('test')
plt.plot(r.step(p['initial']), label='Bruggeman')

r=Hantush()
p=r.set_parameters('test')
plt.plot(r.step(p['initial']), label='Hantush')

plt.legend()
plt.show()