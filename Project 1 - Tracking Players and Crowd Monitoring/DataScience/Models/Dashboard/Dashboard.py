import panel as pn

import os
import sys
import matplotlib.pyplot as plt

'''
This is the Dashboard class that will be used to create the dashboard. It has the following functions:
add_plot: Adds a plot to the dashboard
add_widget: Adds a widget to the dashboard
add_detail: Adds a detail to the dashboard
construct_dashboard: Constructs the dashboard
show: Displays the dashboard


The dashboard class can be used in conjuction with other modules to create a dashboard. For example, the dashboard can be used with the MQTTManager to display the data received from the MQTT broker. The dashboard can also be used with the ContactTracer to display the contacts of the user based on the time and location
'''

sys.path.append(r'e:\\Dev\\Deakin\\redbackoperations-T2_2023\\Project 1 - Tracking Players and Crowd Monitoring\\DataScience\\Models')


from Collision_Prediction import Co_Pred
from Contact_Tracing import Tracer
from HeartRate_Monitoring import HeartRateMonitor
# from Individual_Tracking import Tracker_run

from Overcrowding_Detection import ODHG


Collision_Panel = Co_Pred.plot_panel
Contact_Panel = Tracer.plot_panel
HeartRate_Panel = HeartRateMonitor.plot_panel
# Individual_Panel = Tracker_run.plot_panel()
Overcrowding_Panel = ODHG.plot_panel

class Dashboard:
    def __init__(self, plots=None, widgets=None, details=None):
        self.plots = plots if plots else []
        self.widgets = widgets if widgets else []
        self.details = details if details else []
       

    def save_plot_as_image(self, plot, img_name):
        """Save the provided Matplotlib figure or Panel Matplotlib pane as an image and return a PNG pane."""
        # Check if the plot is a Panel Matplotlib pane and extract the figure if so
        if isinstance(plot, pn.pane.Matplotlib):
            fig = plot.object
        elif isinstance(plot, plt.Axes):  # Check if it's an AxesSubplot object
            fig = plot.figure
        else:
            fig = plot

     # Assuming you save images in an 'images' directory
        images=r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\Dashboard\images'
        images_dir = os.path.join(images, img_name)
        fig.savefig(images_dir)
        image_pane = pn.pane.PNG(images_dir, width=500)
        return image_pane
    def add_plot(self, fig, img_name):
        """Add a plot to the dashboard by saving it as an image."""
        image_pane = self.save_plot_as_image(fig, img_name)
        self.plots.append(image_pane)

    def add_gif_to_dashboard(self, gif_path):
        """
        Add a GIF to the Panel dashboard.

        Parameters:
        - gif_path (str): Path to the GIF file.
        """
        
        gif_pane = pn.pane.Image(gif_path, width=500)
        self.plots.append(gif_pane)
        
    def add_widget(self, widget):
        """Add a widget to the dashboard."""
        if widget:
            self.widgets.append(widget)

    def add_detail(self, detail):
        """Add a detail (like text or HTML) to the dashboard."""
        if detail:
            self.details.append(detail)

    def construct_dashboard(self):
        """Construct and return the dashboard layout by interleaving plots and details."""
        items = []
        
        # Adding widgets if any
        if self.widgets:
            items.append(pn.Row(*self.widgets))
        
        # Interleave plots and details
        for plot, detail in zip(self.plots, self.details):
            items.append(plot)
            items.append(detail)
        
        # If there are any remaining plots or details, add them as well
        remaining_plots = len(self.plots) - len(self.details)
        for i in range(remaining_plots):
            items.append(self.plots[len(self.details) + i])

        remaining_details = len(self.details) - len(self.plots)
        for i in range(remaining_details):
            items.append(self.details[len(self.plots) + i])
        
        dashboard = pn.Column(*items)
        return dashboard

    def show(self):
        """Display the dashboard."""
        dashboard = self.construct_dashboard()
        dashboard.show()

# Create the Dashboard instance and add plots
dashboard_instance = Dashboard()

dashboard_instance.add_gif_to_dashboard(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\Dashboard\images\Collision_P.gif')
coll_deets='The above plot provides insights into potential collisions among tracked users over a specified timeframe. Each colored line represents the predicted trajectory of a different user, with the starting point marked as -Start- and the predicted endpoint as -End-. The presence of a red circle indicates a predicted point of collision. The size of the circle is proportional to the likelihood of a collision, with larger circles signifying higher probabilities. Such predictions can be instrumental in ensuring user safety by preempting and averting possible collisions.'
dashboard_instance.add_detail(coll_deets)
dashboard_instance.add_plot(Tracer.plot_panel, "Contact_Tracing.png")
cont_deets='The plot above is a visual representation of user contacts based on temporal and spatial data. It traces the interactions between different users, highlighting moments of close proximity. Such visualizations are crucial, especially in scenarios like infectious disease outbreaks, where understanding person-to-person contact patterns can play a pivotal role in containment and mitigation strategies.'
dashboard_instance.add_detail(cont_deets)
dashboard_instance.add_plot(HeartRateMonitor.plot_panel, "HeartRate_Monitoring.png")
HR_deets='The displayed plot captures the heart rate data of users over time, providing a continuous monitor of their cardiovascular health. Fluctuations, spikes, or irregular patterns can be indicative of underlying health conditions or moments of heightened stress or activity. Regular monitoring can aid in timely interventions and ensure the well-being of users.'
dashboard_instance.add_detail(HR_deets)
# dashboard_instance.add_plot(Tracker_run.plot_panel)
dashboard_instance.add_plot(ODHG.plot_panel, "Overcrowding_Detection.png")
dashboard_instance.add_plot(ODHG.plot_panel2, "Overcrowding_Detection2.png")
OD_deets='The presented visualizations delve into crowd density estimation within a specified area. The first plot delineates clusters, showcasing the congregation of individuals in specific zones. The subsequent heatmap offers a gradient view of crowd density, with warmer colors signifying areas of higher congestion. Such depictions are invaluable in scenarios that demand crowd management, ensuring safety protocols, and optimizing space utilization.'
dashboard_instance.add_detail(OD_deets)
# Construct and save the dashboard
constructed_dashboard = dashboard_instance.construct_dashboard()
constructed_dashboard.save(r'E:\Dev\Deakin\redbackoperations-T2_2023\Project 1 - Tracking Players and Crowd Monitoring\DataScience\Models\Dashboard\dashboard.html')
