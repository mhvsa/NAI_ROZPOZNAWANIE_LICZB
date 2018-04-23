import tkinter as tki

from numpy import matrix

import neural_network as network

root = tki.Tk()

frm = tki.Frame(root, bd=16, relief='sunken')
frm.grid()

number_vars = []
radio_buttons = []

for i in range(24):
    number_vars.append(tki.BooleanVar())

nv = 0

for i in range(6):
    for j in range(4):
        radio_buttons.append(tki.Radiobutton(frm, variable=number_vars[nv], value=True))
        radio_buttons[nv].config(indicatoron=0, bd=4, width=4)
        radio_buttons[nv].grid(row=i, column=j)
        nv += 1


def deselect_all():
    init_radio_buttons()


def init_radio_buttons():
    number_vars.clear()
    radio_buttons.clear()
    nv = 0
    for i in range(24):
        number_vars.append(tki.BooleanVar())
    for i in range(6):
        for j in range(4):
            radio_buttons.append(tki.Radiobutton(frm, variable=number_vars[nv], value=True))
            radio_buttons[nv].config(indicatoron=0, bd=4, width=4)
            radio_buttons[nv].grid(row=i, column=j)
            nv += 1
    root.mainloop()


def get_values():
    print(list(map(lambda val: 1 if val.get() else 0,number_vars)))

def test_input():
    network.getAnswer(matrix(list(map(lambda val: 1 if val.get() else 0,number_vars))).transpose())


tki.Button(frm, command=deselect_all, text="rst").grid(row=7, column=0)
tki.Button(frm, command=test_input, text="val").grid(row=7, column=1)
tki.Button(frm, command=get_values, text="get").grid(row=7, column=2)

print("===== SIEĆ URUCHOMIONA ====")

root.mainloop()
