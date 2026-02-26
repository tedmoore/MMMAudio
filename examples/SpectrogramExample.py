import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmm_python import *
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PySide6.QtCore import Signal, QObject, Qt
from PySide6.QtGui import QBrush, QPen, QColor


class SignalEmitter(QObject):
    """Helper class to emit signals from the audio callback thread."""
    data_ready = Signal(object)

class SpectrogramGraphicsView(QGraphicsView):
    """Graphics view that displays magnitude data as bars."""
    
    def __init__(self, num_bars=513, parent=None):
        super().__init__(parent)
        self.num_bars = num_bars
        self.bar_height = 400
        
        # Create scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Set up view properties
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHint(self.renderHints().Antialiasing, False)  # Disable for performance
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
        self.setOptimizationFlags(QGraphicsView.DontSavePainterState)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        
        # Create bar items
        self.bars = []
        self.bar_width = 800 / num_bars
        brush = QBrush(QColor(255, 255, 255))
        pen = QPen(Qt.NoPen)
        
        for i in range(num_bars):
            bar = QGraphicsRectItem(0, 0, self.bar_width, 0)
            bar.setBrush(brush)
            bar.setPen(pen)
            bar.setPos(i * self.bar_width, self.bar_height)
            self.scene.addItem(bar)
            self.bars.append(bar)
        
        # Set scene rect
        self.scene.setSceneRect(0, 0, 800, self.bar_height)
        self.setMinimumSize(800, 400)
    
    def update_data(self, data):
        """Update the bar heights from magnitude data."""
        if isinstance(data, np.ndarray):
            magnitudes = data
        else:
            magnitudes = np.array(data)
        
        # Update each bar's height and position
        for i, mag in enumerate(magnitudes[:self.num_bars]):
            # Normalize magnitude
            normalized = mag / 15.0
            bar_height = normalized * self.bar_height
            
            # Update bar geometry (grows upward from bottom)
            self.bars[i].setRect(0, 0, self.bar_width, bar_height)
            self.bars[i].setPos(i * self.bar_width, self.bar_height - bar_height)
    
    def resizeEvent(self, event):
        """Handle resize to scale the view."""
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.IgnoreAspectRatio)


class SpectrogramWindow(QMainWindow):
    """Main window containing the spectrogram display."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Spectrogram")
        
        # Create the graphics view widget
        self.spectrogram_widget = SpectrogramGraphicsView(num_bars=513)
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