import sys
import traceback
import gi
from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg as FigureCanvas
from matplotlib.backends.backend_gtk4 import NavigationToolbar2GTK4 as NavigationToolbar
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp
gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gio
import numpy as np

class VariableDefinitionWidget(Gtk.Box):
    def __init__(self, window):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL)
        self.window = window
        self.name_entry = Gtk.Entry()
        self.derivative_label = Gtk.Label(label="d<>/dt =")
        self.derivative_entry = Gtk.Entry()
        self.initial_value_entry = Gtk.Entry()
        self.remove_button = Gtk.Button(label="remove")
        self.remove_button.connect("clicked", self.remove_variable)

        self.append(self.name_entry)
        self.append(self.derivative_label)
        self.append(self.derivative_entry)
        self.append(Gtk.Label(label="intial value:"))
        self.append(self.initial_value_entry)
        self.append(self.remove_button)

    def remove_variable(self, widget):
        self.window.remove_variable(self)

    def get_data(self):
        return (self.name_entry.get_text(), self.derivative_entry.get_text(),
                self.initial_value_entry.get_text())

class ParameterDefinitionWidget(Gtk.Box):
    def __init__(self, window):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL)
        self.window = window
        self.name_entry = Gtk.Entry()
        self.value_entry = Gtk.Entry()
        self.remove_button = Gtk.Button(label="remove")
        self.remove_button.connect("clicked", self.remove_parameter)
        # TODO: add slider with changeable bounds

        self.append(self.name_entry)
        self.append(Gtk.Label(label="="))
        self.append(self.value_entry)
        self.append(self.remove_button)

    def remove_parameter(self, widget):
        self.window.remove_parameter(self)

    def get_data(self):
        return (self.name_entry.get_text(), self.value_entry.get_text())

class AppWindow(Gtk.Window):
    def __init__(self, app):
        # window
        super().__init__(application=app)
        self.set_default_size(600, 250)
        self.set_title("ODE Studio")

        # top
        self.independent_variable_entry = Gtk.Entry()
        self.independent_variable_start_entry = Gtk.Entry()
        self.independent_variable_end_entry = Gtk.Entry()
        self.samples_entry = Gtk.Entry()
        self.run_button = Gtk.Button(label="Run!")
        self.run_button.connect("clicked", self.run)

        self.top_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.top_box.append(Gtk.Label(label="Independent Variable:"))
        self.top_box.append(self.independent_variable_entry)
        self.top_box.append(Gtk.Label(label="="))
        self.top_box.append(self.independent_variable_start_entry)
        self.top_box.append(Gtk.Label(label=".."))
        self.top_box.append(self.independent_variable_end_entry)
        self.top_box.append(Gtk.Label(label="samples:"))
        self.top_box.append(self.samples_entry)
        self.top_box.append(self.run_button)

        # variable/equations/parameter definitions
        self.add_new_variable_button = Gtk.Button(label="add new variable")
        self.add_new_variable_button.connect("clicked", self.add_new_variable)
        self.variable_definitions = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.variable_definitions.append(self.add_new_variable_button)

        self.add_new_parameter_button = Gtk.Button(label="add new parameter")
        self.add_new_parameter_button.connect("clicked", self.add_new_parameter)
        self.parameter_definitions = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.parameter_definitions.append(self.add_new_parameter_button)

        self.variable_and_parameter_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.variable_and_parameter_box.append(Gtk.Label(label="Variables:"))
        self.variable_and_parameter_box.append(self.variable_definitions)
        self.variable_and_parameter_box.append(Gtk.Label(label="Parameter:"))
        self.variable_and_parameter_box.append(self.parameter_definitions)

        # plotting
        self.xaxis_selection_combo_box = Gtk.ComboBoxText()
        self.yaxis_selection_combo_box = Gtk.ComboBoxText()
        self.xaxis_selection_combo_box.connect("changed", self.plot)
        self.yaxis_selection_combo_box.connect("changed", self.plot)

        self.axis_choices_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.axis_choices_box.append(self.xaxis_selection_combo_box)
        self.axis_choices_box.append(self.yaxis_selection_combo_box)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.data = None
        self.plot_canvas = FigureCanvas(self.fig)  # a Gtk.DrawingArea
        self.plot_canvas.set_hexpand(True)
        self.plot_canvas.set_vexpand(True)

        self.plot_toolbar = NavigationToolbar(self.plot_canvas)

        self.plot_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.plot_box.append(self.axis_choices_box)
        self.plot_box.append(self.plot_canvas)
        self.plot_box.append(self.plot_toolbar)

        # putting it together
        self.hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.hbox.append(self.variable_and_parameter_box)
        self.hbox.append(self.plot_box)

        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.vbox.append(self.top_box)
        self.vbox.append(self.hbox)

        self.set_child(self.vbox)

    def add_new_variable(self, widget):
        self.variable_definitions.append(VariableDefinitionWidget(self))

    def add_new_parameter(self, widget):
        self.parameter_definitions.append(ParameterDefinitionWidget(self))

    def remove_variable(self, widget):
        self.variable_definitions.remove(widget)

    def remove_parameter(self, widget):
        self.parameter_definitions.remove(widget)

    def run(self, widget):
        try:
            # get data
            independent_variable_name = self.independent_variable_entry.get_text()
            independent_variable_start = float(self.independent_variable_start_entry.get_text())
            independent_variable_end = float(self.independent_variable_end_entry.get_text())
            nsamples = int(self.samples_entry.get_text())

            equations = [v.get_data()
                    for v in self.variable_definitions if not isinstance(v, Gtk.Button)]
            parameters = [p.get_data()
                    for p in self.parameter_definitions if not isinstance(p, Gtk.Button)]

            # parse data
            if len(equations) == 0:
                return
            dependent_var_names, rhs_expressions, initial_values = zip(*equations)
            if len(parameters) == 0:
                parameter_names = parameter_values = [] # it doenst matter that these are the same object
            else:
                parameter_names, parameter_values = zip(*parameters)
            expression = "(" + ",".join(rhs_expressions) + ")"
            initial_values = list(map(float, initial_values))
            parameter_values = list(map(float, parameter_values))

            def rhs(independent_var_value, dependent_var_values, *parameter_values):
                var_mapping = dict(zip(dependent_var_names, dependent_var_values))
                var_mapping[independent_variable_name] = independent_var_value
                var_mapping.update(dict(zip(parameter_names, parameter_values)))
                g = globals()
                g.update(vars(np))
                return eval(expression, var_mapping, g)

            # solve ode
            sol = solve_ivp(rhs,
                    (independent_variable_start, independent_variable_end),
                    initial_values,
                    args=parameter_values)

            self.data = dict(zip(dependent_var_names, sol.y))
            self.data[independent_variable_name] = sol.t

            # update axis choices
            names = dependent_var_names + (independent_variable_name,)
            self.xaxis_selection_combo_box.remove_all()
            self.yaxis_selection_combo_box.remove_all()
            for name in names:
                self.xaxis_selection_combo_box.append_text(name)
                self.yaxis_selection_combo_box.append_text(name)

        except Exception as e:
            traceback.print_exception(e)

    def plot(self, widget):
        _ = widget
        try:
            xaxis = self.xaxis_selection_combo_box.get_active_text()
            yaxis = self.yaxis_selection_combo_box.get_active_text()
            if xaxis is None or yaxis is None:
                return
            x = self.data[xaxis]
            y = self.data[yaxis]
            self.ax.clear()
            self.ax.plot(x, y)
            self.ax.set_xlabel(xaxis)
            self.ax.set_ylabel(yaxis)
            self.fig.canvas.draw()

        except Exception as e:
            traceback.print_exception(e)

class App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.anna-jana.ODEStudio")
        GLib.set_application_name("ODE Studio")

    def do_activate(self):
        window = AppWindow(self)
        #window.present()
        window.show()

if __name__ == "__main__":
    app = App()
    exit_status = app.run(sys.argv)
    sys.exit(exit_status)
