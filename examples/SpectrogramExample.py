import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmm_python import *
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import Signal, QObject, Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen


class SignalEmitter(QObject):
    """Helper class to emit signals from the audio callback thread."""
    data_ready = Signal(object)

class SpectrogramWidget(QWidget):
    """Widget that displays magnitude data as a bar graph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.magnitudes = np.zeros(513)
        self.setMinimumSize(800, 400)
        self.color = QColor(255, 255, 255)

        
    def update_data(self, data):
        """Update the magnitude data and trigger a repaint."""
        if isinstance(data, np.ndarray):
            self.magnitudes = data
        else:
            self.magnitudes = np.array(data)
        self.update()  # Trigger paintEvent
        
    def paintEvent(self, event):
        """Draw the bar graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        # Calculate bar width
        num_bars = len(self.magnitudes)
        bar_width = width / num_bars
                
        # Draw bars
        for i, mag in enumerate(self.magnitudes):
            # Normalize to 0-1 range
            normalized = mag / 15.0
            
            # Calculate bar height (inverted so high values are at top)
            bar_height = normalized * height
            
            painter.setPen(QPen(self.color, max(bar_width, 1)))
            painter.setBrush(self.color)
            
            # Draw bar from bottom to top
            x = i * bar_width
            y = height - bar_height
            painter.drawRect(int(x), int(y), max(int(bar_width), 1), int(bar_height))


class SpectrogramWindow(QMainWindow):
    """Main window containing the spectrogram display."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Spectrogram")
        
        # Create the spectrogram widget
        self.spectrogram_widget = SpectrogramWidget()
        self.setCentralWidget(self.spectrogram_widget)
        
        # Create signal emitter for thread-safe updates
        self.signal_emitter = SignalEmitter()
        self.signal_emitter.data_ready.connect(self.spectrogram_widget.update_data)
        
    def callback(self, args):
        """Callback function to receive magnitude data."""
        # Emit signal to update GUI in main thread
        self.signal_emitter.data_ready.emit(args)
    
    def closeEvent(self, event):
        """Handle window close event."""
        QApplication.quit()
        event.accept()

if __name__ == "__main__":
    
    app = QApplication()
    
    window = SpectrogramWindow()
    window.show()
    
    m = MMMAudio(128, graph_name="SpectrogramExample", package_name="examples")
    m.register_callback("mags", window.callback)
    m.start_audio()

    def shutdown():
        m.stop_audio()
        m.stop_process()

    app.aboutToQuit.connect(shutdown)
    
    sys.exit(app.exec())