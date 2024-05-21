import tkinter as tk
from tkinter import END, filedialog
import os
from trulens_eval.tru import Tru
from threading import Thread


class DashboardLauncher(tk.Tk):
    def __init__(self):
        self.trulens_object = None
        self.init_ui()

    def open_file_manager(self):
        filepath = filedialog.askopenfilename(filetypes=[("SQLite files", "*.sqlite")])
        self.file_path_entry.delete(0, tk.END)
        self.file_path_entry.insert(0, filepath)

    def launch_dashboard_thread(self):
        launch_thread = Thread(target=self.launch_dashboard)
        launch_thread.start()

    def launch_dashboard(self):
        self.close_button["state"] = tk.DISABLED
        self.launch_button["state"] = tk.DISABLED
        db_path = self.file_path_entry.get()
        if not db_path:
            self.status_label.config(text="Select a path", fg="red")
            self.launch_button["state"] = tk.NORMAL
            self.close_button["state"] = tk.NORMAL
            return
        if not os.path.isfile(db_path):
            self.status_label.config(text="Invalid file path", fg="red")
            self.launch_button["state"] = tk.NORMAL
            self.close_button["state"] = tk.NORMAL
            return
        self.status_label.config(text="Launching Dashboard...", fg="green")
        self.trulens_object = Tru(database_file=db_path)

        try:
            addr = self.address.get()
            prt = int(self.port.get())
            self.trulens_object.run_dashboard(port=prt)
        except Exception as e:
            self.status_label.config(text=str(e), fg="red")
            self.launch_button["state"] = tk.NORMAL
        else:
            self.status_label.config(text="Dashboard Launched", fg="green")
            url = f"http://{addr}:{prt}/"
            os.system(f"start {url}")

        self.close_button["state"] = tk.NORMAL

    def close_dashboard(self):
        if self.trulens_object:
            if self.trulens_object._dashboard_proc:
                self.status_label.config(text="Closing Dashboard...", fg="green")
                self.trulens_object.stop_dashboard()
                self.status_label.config(text="Dashboard Closed", fg="green")
            else:
                self.status_label.config(text="Dashboard not launched yet", fg="red")
        else:
            self.status_label.config(text="Dashboard not launched yet", fg="red")
        self.launch_button["state"] = tk.NORMAL

    def on_closing(self):
        """Safe exit for the application"""
        if self.trulens_object:
            if self.trulens_object._dashboard_proc:
                self.trulens_object.stop_dashboard(force=True)
        self.destroy()

    def init_ui(self):
        self.geometry("800x200")
        self.title("TruLens Dashboard Launcher")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.frame = tk.Frame(self)
        self.frame.pack(expand=True, anchor="center")

        self.file_path_label = tk.Label(self.frame, text="File Path:", width=10)
        self.file_path_label.grid(row=0, column=0)

        self.file_path_entry = tk.Entry(self.frame, width=100)
        self.file_path_entry.grid(row=0, column=1)

        self.file_path_button = tk.Button(
            self.frame, text="Open", command=self.open_file_manager, width=10, padx=5
        )
        self.file_path_button.grid(row=0, column=2)

        self.blank_space_1 = tk.Label(self, text="", width=10)
        self.blank_space_1.pack()

        self.frame3 = tk.Frame(self)
        self.frame3.pack(expand=True, anchor="center")

        self.address_label = tk.Label(self.frame3, text="Address: ", width=10)
        self.address_label.grid(row=0, column=0)

        self.address = tk.Entry(self.frame3, width=20)
        self.address.insert(END, "localhost")
        self.address.grid(row=0, column=1)
        self.address.config(state="disabled")

        self.port_label = tk.Label(self.frame3, text="Port: ", width=10)
        self.port_label.grid(row=1, column=0)

        self.port = tk.Entry(self.frame3, width=20)
        self.port.insert(END, "8501")
        self.port.grid(row=1, column=1)

        self.blank_space_2 = tk.Label(self, text="", width=10)
        self.blank_space_2.pack()

        self.frame2 = tk.Frame(self)
        self.frame2.pack(expand=True, anchor="center")

        self.launch_button = tk.Button(
            self.frame2, text="Launch Dashboard", command=self.launch_dashboard_thread
        )
        self.launch_button.grid(row=0, column=0)

        self.blank_label = tk.Label(self.frame2, text="", width=10)
        self.blank_label.grid(row=0, column=1)

        self.close_button = tk.Button(
            self.frame2, text="Close Dashboard", command=self.close_dashboard
        )
        self.close_button.grid(row=0, column=2)

        self.status_label = tk.Label(self, text="", width=500)
        self.status_label.pack(expand=True, anchor="center")


def main():
    dashboard_launcher = DashboardLauncher()
    dashboard_launcher.mainloop()
