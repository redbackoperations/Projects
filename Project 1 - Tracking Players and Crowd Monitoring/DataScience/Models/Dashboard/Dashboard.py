import panel as pn

class Dashboard:
    def __init__(self):
        self.plots = []
        self.widgets = []
        self.details = []
    
    def add_plot(self, plot):
        """Add a plot to the dashboard."""
        self.plots.append(plot)

    def add_widget(self, widget):
        """Add a widget to the dashboard."""
        self.widgets.append(widget)

    def add_detail(self, detail):
        """Add a detail (like text or HTML) to the dashboard."""
        self.details.append(detail)

    def construct_dashboard(self):
        """Construct and return the dashboard layout."""
        dashboard = pn.Column(
            pn.Row(*self.widgets),  # Place widgets at the top
            pn.Row(*self.details),  # Details below widgets
            pn.Row(*self.plots)     # Plots at the bottom or you can arrange as needed
        )
        return dashboard

    def show(self):
        """Display the dashboard."""
        dashboard = self.construct_dashboard()
        dashboard.show()
