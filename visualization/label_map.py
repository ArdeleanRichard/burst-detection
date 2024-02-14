LABEL_COLOR_MAP = {-1: 'gray',
                   0: 'white',
                   1: 'red',
                   2: 'blue',
                   3: 'green',
                   4: 'black',
                   5: 'yellow',
                   6: 'cyan',
                   7: 'magenta',
                   8: 'tab:purple',
                   9: 'tab:orange',
                   10: 'tab:brown',
                   11: 'tab:pink',
                   12: 'lime',
                   13: 'orchid',
                   14: 'khaki',
                   15: 'lightgreen',
                   16: 'orangered',
                   17: 'salmon',
                   18: 'silver',
                   19: 'yellowgreen',
                   20: 'royalblue',
                   21: 'beige',
                   22: 'crimson',
                   23: 'indigo',
                   24: 'darkblue',
                   25: 'gold',
                   26: 'ivory',
                   27: 'lavender',
                   28: 'lightblue',
                   29: 'olive',
                   30: 'sienna',
                   31: 'salmon',
                   32: 'teal',
                   33: 'turquoise',
                   34: 'wheat',
                   }

LABEL_COLOR_MAP_SMALLER = {
                   0: 'red',
                   1: 'blue',
                   2: 'green',
                   3: 'yellow',
                   4: 'cyan',
                   5: 'magenta',
                   }


MARKERS = ['o', 'v', '<', '>', '^', '1', '2', '3', '4', "8", "s", "p", "P", "*", "h", "H", "X", "D", "d", "1", "2", "3", "4", 4, 5, 6, 7]
COLORS = []
LABEL_NAMES = {}
for idx, mrk in enumerate(MARKERS):
    COLORS.append(f'C{idx}')
    LABEL_NAMES[idx] = f'C{idx}'