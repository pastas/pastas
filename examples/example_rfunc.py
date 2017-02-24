from pastas.rfunc import *
import matplotlib.pyplot as plt

r=Gamma()
p=r.set_parameters('test')
plt.plot(r.step(p['initial']), label='Gamma')

r=Exponential()
p=r.set_parameters('test')
plt.plot(r.step(p['initial']), label='Exponential', linestyle='--', lw=2)

r=Hantush(up=True)
p=r.set_parameters('test')
plt.plot(r.step(p['initial']), label='Hantush')

r=Bruggeman()
p=r.set_parameters('test')
plt.plot(r.step(p['initial']), label='Bruggeman')

plt.legend()
plt.show()