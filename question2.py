import random
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Assumes that the world is built on 20*20
ROW_NUM = 20
COLUMN_NUM = 20


# class of cell in my automat
class Cell(object):
    def __init__(self, area, temp, height, wind_power, direction, aqi):
        self._area = area
        self._temp = temp   # temperature
        self._height = height
        self._wind_power = wind_power
        self._direction = direction
        self._aqi = aqi     # scale of pollution(AQI= Air Quality Index)
        # 1 = has cloud no rain, 2 = has cloud with rain, 3 = don't have cloud (different probability to any option)
        self._cloud = random.choices([1, 2, 3], weights=[0.2, 0.05, 0.75], k=1)[0]

    def get_area(self):
        return self._area

    def get_temp(self):
        return self._temp

    def get_height(self):
        return self._height

    def get_wind_power(self):
        return self._wind_power

    def get_direction(self):
        return self._direction

    def get_aqi(self):
        return self._aqi

    def get_cloud(self):
        return self._cloud

    def set_area(self, area):
        self._area = area

    def set_temp(self, temp):
        self._temp = temp

    def set_height(self, height):
        self._height = height

    def set_wind_power(self, wind_power):
        self._wind_power = wind_power

    def set_direction(self, direction):
        self._direction = direction

    def set_aqi(self, aqi):
        self._aqi = aqi

    def set_cloud(self, cloud):
        self._cloud = cloud

    # return different color to different area as I choose
    def area_color(self):
        color = {
            'S': "royalblue", 'L': "sienna", 'I': "turquoise", 'F': "green", 'C': "gold", 'B': 'red'
        }
        if type(self._area) is tuple:
            return color[self._area[0]]
        return color[self._area]

    # return the color of the cloud by the number he has
    def cloud_color(self):
        color = {
            1: "white", 2: "grey", 3: None
        }
        return color[self._cloud]

    # override
    def __str__(self):
        # unicode of arrow
        switch_compass_symbol = {
            'N': '\u2191', 'NE': '\u2197', 'NW': '\u2196', 'E': '\u2192', 'W': '\u2190', 'S': '\u2193',
            'SE': '\u2198', 'SW': '\u2199'
        }
        # u2103 is unicode of image sign of degrees
        text = "{} {:.1f}\u2103\n P:{:.0f}".format(switch_compass_symbol[self._direction], self._temp, self._aqi)
        return text


# build a map like I get in the file 'word.dat'
def build_map(file_name, star_pollution):
    map_matrix = []
    line_num = 0
    with open(file_name, 'r') as file:
        while True:
            line = file.readline()  # Read one line at a time
            if not line:  # If line is empty, EOF is reached
                break
            else:
                for index in range(len(line)):
                    if index == 0:
                        map_matrix.append([line[index]])
                    else:
                        if line[index] != '\n':
                            map_matrix[line_num].append(line[index])
            line_num += 1
    init_val(map_matrix, star_pollution)
    return map_matrix


# I choose initial value for any area
def init_val(map_matrix, star_pollution):
    # S-sea, L- land, I- iceberg, F- forest
    switch_temp = {
        'S': 20, 'L': 28, 'I': -20, 'F': 15, 'C': 25
    }
    switch_height = {
        'S': 0, 'L': 50, 'I': 20, 'F': 150, 'C': 300
    }
    # Beaufort scale
    switch_wind = {
        'S': 4, 'L': 2, 'I': 3, 'F': 0, 'C': 1
    }
    # I gave initial values to make the map as realistic as possible
    for row in range(ROW_NUM):
        for col in range(COLUMN_NUM):
            temp = switch_temp.get(map_matrix[row][col])
            height = switch_height.get(map_matrix[row][col])
            if map_matrix[row][col] != 'S':
                height = random.choice([height, height + height/2, height + 50])
            wind_power = switch_wind.get(map_matrix[row][col])
            aqi = star_pollution
            direction = random.choice(['S', 'N', 'W', 'E', 'SE', 'SW', 'NE', 'NW'])     # south,north, west,east,etc
            map_matrix[row][col] = Cell(map_matrix[row][col], temp, height, wind_power, direction, aqi)


# check how much clouds has in my map
def count_clouds(update_map):
    count = 0
    for row in update_map:
        for cell in row:
            if cell.get_cloud() != 3:  # 3 means no cloud
                count += 1
    return count


# return how much rain probability has in this area
# At cold area has more probability to rain
def rain_probability(update_map, row, col):
    cell = update_map[row][col]
    cell_temp = cell.get_temp()
    if cell_temp <= 10:
        return 0.5
    elif 10 < cell_temp <= 20:
        return 0.4
    elif 20 < cell_temp <= 30:
        return 0.2
    elif 30 < cell_temp <= 35:
        return 0.1
    else:
        return 0.05


# maintain the number of clouds in my world
def maintain_clouds(update_map, target_cloud):
    current_cloud_count = count_clouds(update_map)

    if current_cloud_count > target_cloud:
        # Too many clouds: remove some clouds randomly
        cells_with_clouds = [(r, c) for r in range(ROW_NUM) for c in range(COLUMN_NUM) if update_map[r][c].get_cloud() != 3]
        cells_to_remove = random.sample(cells_with_clouds, current_cloud_count - target_cloud)
        for r, c in cells_to_remove:
            update_map[r][c].set_cloud(3)  # Set to no cloud
    elif current_cloud_count < target_cloud:
        # Too few clouds: add clouds randomly
        empty_cells = [(r, c) for r in range(ROW_NUM) for c in range(COLUMN_NUM) if update_map[r][c].get_cloud() == 3]
        cells_to_add = random.sample(empty_cells, target_cloud - current_cloud_count)
        for r, c in cells_to_add:
            probability = rain_probability(update_map, r, c)
            update_map[r][c].set_cloud(random.choices([1, 2], weights=[1 - probability, probability], k=1)[0])


# create the canvas as matrix that in any cell define as rectangle,text,oval
def create_canvas(canvas_details):
    canvas_cells = [[None for _ in range(ROW_NUM)] for _ in range(COLUMN_NUM)]   # create empty matrix
    cg = 0.15  # cloud_gap
    size_cell = 45
    for row in range(ROW_NUM):
        for col in range(COLUMN_NUM):
            next_col = col + 1
            next_row = row + 1
            # the number here make the text be in the right place in the cell
            canvas_square_id = canvas_details.create_rectangle(col*size_cell, row*size_cell,
                                                               next_col*size_cell, next_row*size_cell)
            canvas_text_id = canvas_details.create_text((col+0.5) * size_cell,
                                                        (row+0.30) * size_cell, font="Ariel 8 bold")
            canvas_cloud_id = canvas_details.create_oval((col+cg)*size_cell, (row+0.5+cg)*size_cell,
                                                         (next_col-cg)*size_cell, (next_row-cg)*size_cell, width=0)
            canvas_cells[row][col] = (canvas_square_id, canvas_text_id, canvas_cloud_id)
    return canvas_cells


# update the cells by they attribute
def update_canvas(update_map, canvas_cells, canvas):
    for row in range(ROW_NUM):
        for col in range(COLUMN_NUM):
            cell = update_map[row][col]
            canvas_cell_text = cell
            canvas_cell_color = cell.area_color()
            canvas_cell_cloud = cell.cloud_color()

            (canvas_square_id, canvas_text_id, canvas_cloud_id) = canvas_cells[row][col]
            canvas.itemconfig(canvas_square_id, fill=canvas_cell_color)
            canvas.itemconfig(canvas_text_id, text=canvas_cell_text)
            if canvas_cell_cloud is not None:
                canvas.itemconfig(canvas_cloud_id, fill=canvas_cell_cloud)
            else:
                canvas.itemconfig(canvas_cloud_id, fill="")


# next temperature of the cell by the transition rule
def next_temp(cell, av_pollution):
    updated_temp = cell.get_temp()
    # By trial and error, we choose numbers that maintain a balanced system throughout a year
    # when the initial pollution is 0
    probability = min(cell.get_aqi() / 1000 + av_pollution * 0.008, 1)
    if random.choices([True, False], weights=[probability, 1 - probability])[0]:
        updated_temp = updated_temp + random.choices([0.1, 0.25], weights=[probability, 1 - probability])[0]
    return round(updated_temp, 2)


# next aqi(Air Quality Index ) of the cell by the transition rule
def next_aqi(cell, save_aqi, row, col):
    switch_pollution = {
        0: 0, 1: 0.05, 2: 0.15, 3: 0.25, 4: 0.35, 5: 0.45, 6: 0.55, 7: 0.65,
        8: 0.80, 9: 0.85, 10: 0.90, 11: 0.95, 12: 0.98
    }
    switch_direction = {
        'SE': (1, 1), 'SW': (1, -1), 'S': (1, 0), 'NW': (-1, -1), 'NE': (-1, 1), 'N': (-1, 0),
        'E': (0, 1), 'W': (-1, 0)
    }
    add_pollution_to = switch_direction[cell.get_direction()]
    city_add = 0
    if cell.get_area() == 'C':
        city_add = 1    # every day the city create 1 pollution
    to_remove = switch_pollution[cell.get_wind_power()]*cell.get_aqi()
    save_aqi[(row + add_pollution_to[0]) % ROW_NUM][(col + add_pollution_to[1]) % COLUMN_NUM] += to_remove
    return round(cell.get_aqi() - to_remove + city_add, 2)


# next wind power and cloud of the cell by the transition rule
def next_wind_power_and_cloud(update_map, row, col):
    neighbors = [
        (-1, -1, 'SE'), (-1, 0, 'S'), (-1, 1, 'SW'),  # Top-left, Top, Top-right
        (0, -1, 'E'),                 (0, 1, 'W'),  # Left,        Right
        (1, -1, 'NE'), (1, 0, 'N'), (1, 1, 'NW')  # Bottom-left, Bottom, Bottom-right
    ]
    max_wind_power = 12  # Beaufort scale
    updated_wind = 0
    probability = rain_probability(update_map, row, col)
    has_neighbor_with_cloud = False
    for ar, ac, cell_direction in neighbors:  # ar = add-row, ac = add col
        neighbor = update_map[(row + ar) % ROW_NUM][(col + ac) % COLUMN_NUM]
        if neighbor.get_direction() == cell_direction:
            updated_wind += neighbor.get_wind_power()
            if neighbor.get_cloud() != 3:   # has cloud
                has_neighbor_with_cloud = True
    if has_neighbor_with_cloud:
        updated_cloud = random.choices([1, 2], weights=[1 - probability, probability], k=1)[0]
    else:
        updated_cloud = 3   # no cloud
    return (min(max_wind_power, max(random.choice([updated_wind - 1, updated_wind + 4, updated_wind + 6]), 0)),
            updated_cloud)


# next direction of the cell by the transition rule
def next_direction(cell):
    clockwise_direction = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    probability = 0.1   # probability to change direction
    option_direction = 8
    update_direction = cell.get_direction()
    if random.choices([True, False], weights=[probability, 1 - probability])[0]:
        update_direction = clockwise_direction[(clockwise_direction.index(cell.get_direction()) + 1) % option_direction]
    return update_direction


# next area of the cell by the transition rule
def next_area(cell):
    update_area = cell.get_area()
    temp = cell.get_temp()
    if type(update_area) is tuple:
        if update_area[1] == 0:
            update_area = 'L'
        else:
            update_area = (update_area[0], update_area[1] - 1)
    elif update_area == 'I' and temp > 0:
        update_area = 'S'   # if iceberg has more than 0 degree he became to sea
    elif update_area == 'F' and temp > 35:
        update_area = ('B', 5)  # if forest get more than 35 degree he burned for 5 days
    return update_area


# set all the variable update
def set_for_all(cell, updated_temp, updated_aqi, updated_wind, update_cloud, update_direction, update_area):
    cell.set_area(update_area)
    cell.set_temp(updated_temp)
    cell.set_wind_power(updated_wind)
    cell.set_cloud(update_cloud)
    cell.set_direction(update_direction)
    cell.set_aqi(updated_aqi)


# choose the next generation of any cell in my world
def next_generation(update_map, target_cloud_count, av_pollution):
    update_after_change = np.empty((ROW_NUM, COLUMN_NUM), dtype=object)  # create empty matrix of objects 20*20
    add_aqi_matrix = np.zeros((ROW_NUM, COLUMN_NUM))
    for row in range(ROW_NUM):
        for col in range(COLUMN_NUM):
            cell = update_map[row][col]
            updated_temp = next_temp(cell, av_pollution)
            updated_aqi = next_aqi(cell, add_aqi_matrix, row, col)
            updated_wind, update_cloud = next_wind_power_and_cloud(update_map, row, col)
            update_direction = next_direction(cell)
            update_area = next_area(cell)
            update_after_change[row][col] = (cell, updated_temp, updated_aqi, updated_wind, update_cloud,
                                             update_direction, update_area)
    add_aqi_matrix = np.round(add_aqi_matrix, 2)
    for row in range(ROW_NUM):
        for col in range(COLUMN_NUM):
            cell, temp, aqi, wind, cloud, direction, area = update_after_change[row][col]
            aqi += add_aqi_matrix[row][col]
            set_for_all(cell, temp, aqi, wind, cloud, direction, area)

    # Ensure the clouds are maintained after updating everything else
    maintain_clouds(update_map, target_cloud_count)


# update the map between the generations
def scheduled_update(my_map, root, my_canvas, lable, sub_lable, days_list, refresh_every_ml,
                     canvas, target_cloud_count, av_pollution):
    next_generation(my_map, target_cloud_count, av_pollution)
    update_canvas(my_map, my_canvas, canvas)

    # Update the generation label
    current_generation = int(lable['text'].split()[1])  # Get the current generation number
    new_generation = current_generation + 1
    lable.config(text="Generation {}".format(new_generation))  # Update label text

    # Update the sub-label with new statistics
    sub_label_text, average_pollution = get_sub_label_text(my_map, days_list)
    sub_lable.config(text=sub_label_text)  # Update the sub label text

    days = 365  # stops the automaton's progress after a number of days
    if new_generation < days:
        root.after(refresh_every_ml, scheduled_update, my_map, root, my_canvas, lable,
                   sub_lable, days_list, refresh_every_ml, canvas, target_cloud_count, average_pollution)
    else:   # create button that destroy the automat
        destroy_button = tk.Button(text="destroy", command=root.destroy, bg="red")
        destroy_button.place(x=10, y=10)


# section c -> write in my sub label the temperature standard deviation, temperature average,
# air pollution standard deviation, air pollution average
def get_sub_label_text(update_map, days_list):
    temp_list = []
    aqi_list = []
    rain = 0
    for i in range(len(update_map)):
        for j in range(len(update_map[i])):
            temp_list.append(update_map[i][j].get_temp())
            aqi_list.append(update_map[i][j].get_aqi())
            if update_map[i][j].get_cloud() == 2:
                rain += 1
    temperature = np.array(temp_list)
    air_pollution = np.array(aqi_list)
    temperature_average = round(np.mean(temperature), 1)
    air_pollution_average = round(np.mean(air_pollution), 2)  # calculate in percentage
    temperature_std = round(np.std(temperature), 1)
    air_pollution_std = round(np.std(air_pollution), 2)

    days_list.append((temperature_average, temperature_std, air_pollution_average, air_pollution_std, rain))

    # u2103  is unicode of --> image of Unicode Character 'DEGREE CELSIUS'
    line1 = "Average temperature: {}\u2103   \t Average air Pollution: {}\n".format(temperature_average,
                                                                                     air_pollution_average)
    line2 = "Standard deviation: {}\u2103 \t\t Standard deviation: {}".format(temperature_std,
                                                                               air_pollution_std)

    return line1 + line2, air_pollution_average


#   create world by the file I put
def create_world(file, star_pollution):
    my_map = build_map(file, star_pollution)
    target_cloud_count = count_clouds(my_map)
    days_list = []
    root = tk.Tk()
    root.title("World Simulation")
    lable = tk.Label(root, text="Generation {}".format(1), font="Ariel")
    sub_label_text, average_pollution = get_sub_label_text(my_map, days_list)
    sub_lable = tk.Label(root, text=sub_label_text)
    refresh_every_ml = 2
    lable.pack()
    sub_lable.pack()

    # size_cell = 45 --> height,width = (ROW_NUM/COLUMN_NUM * size_cell) + 1
    canvas = tk.Canvas(root, height=901, width=901, bg="white")
    canvas.pack()

    my_canvas = create_canvas(canvas)
    update_canvas(my_map, my_canvas, canvas)

    root.after(refresh_every_ml, scheduled_update, my_map, root, my_canvas, lable,
               sub_lable, days_list, refresh_every_ml, canvas, target_cloud_count, average_pollution)
    root.mainloop()
    return days_list


#
def data_of_world(data):
    temp_avg = 0
    temp_std = 1
    poll_avg = 2
    poll_std = 3
    rain_sum = 4
    days_list = np.array(data)
    temperature_throughout_year = np.array(days_list[:, temp_avg])
    temperature_std_throughout_year = np.array(days_list[:, temp_std])
    pollution_throughout_year = np.array(days_list[:, poll_avg])
    pollution_std_throughout_year = np.array(days_list[:, poll_std])
    rain_sum_throughout_year = np.array(days_list[:, rain_sum])

    temp_mean = np.mean(temperature_throughout_year)
    poll_mean = np.mean(pollution_throughout_year)

    # Handle potential division by zero in standardization
    temperature_std_throughout_year_safe = np.where(temperature_std_throughout_year == 0, 1,
                                                    temperature_std_throughout_year)
    pollution_std_throughout_year_safe = np.where(pollution_std_throughout_year == 0, 1, pollution_std_throughout_year)

    # Standardize temperature and pollution arrays
    standardization_temp = (temperature_throughout_year - temp_mean) / temperature_std_throughout_year_safe
    standardization_poll = (pollution_throughout_year - poll_mean) / pollution_std_throughout_year_safe

    correlation_matrix_temp_poll = np.corrcoef(pollution_throughout_year, temperature_throughout_year)
    correlation_temp_poll = correlation_matrix_temp_poll[0, 1]
    correlation_matrix_rain_poll = np.corrcoef(pollution_throughout_year, rain_sum_throughout_year)
    correlation_rain_poll = correlation_matrix_rain_poll[0, 1]

    # Create plot
    plt.plot(standardization_temp, standardization_poll, marker='o', linestyle='-')
    plt.xlabel("Normalized Temperature")
    plt.ylabel("Normalized Pollution")
    plt.title("Standard Scores of Temperature and Pollution Over Time")

    # Display plot
    plt.show()

    return (round(temp_mean, 2), round(poll_mean, 2),
            np.round(correlation_temp_poll, 2), np.round(correlation_rain_poll, 2))


# create cellular automatons with different start pollution in every cell and presents the demanded data
sensitive = []
for pollution in range(0, 25, 5):
    days_data = create_world('word.dat', pollution)
    sensitive.append(data_of_world(days_data))
columns = ["temp mean", "poll mean", "cor-temp,poll", "cor-rain,poll"]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(pd.DataFrame(sensitive, columns=columns))










