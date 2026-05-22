"""
ESPMotion - MQTT Handler Module

Publish-only MQTT client: connects to the broker, publishes a one-shot
info message at startup, and publishes periodic state updates from the
main loop. The firmware never subscribes to inbound topics.

"""
import json
import time
from umqtt.simple import MQTTClient


class MQTTHandler:
    """Publish-only MQTT handler."""

    def __init__(self, config, detector, wlan):
        """
        Initialize MQTT handler

        Args:
            config: Configuration module
            detector: IDetector instance (MVSDetector or MLDetector)
            wlan: WLAN instance
        """
        self.config = config
        self.detector = detector
        self.wlan = wlan
        self.client = None

        # Topics
        self.base_topic = config.MQTT_TOPIC
        self.info_topic = f"{config.MQTT_TOPIC}/info"

        # Publishing state
        self.last_variance = 0.0
        self.last_state = 0  # STATE_IDLE

    # Socket timeout applied before every MQTT op. umqtt.simple internally
    # calls sock.setblocking(True) at the end of wait_msg(), which clears
    # any timeout we set after connect — so we MUST re-apply it each time
    # or publish will block forever if lwIP gets jammed.
    _MQTT_OP_TIMEOUT_S = 3

    def _apply_op_timeout(self):
        try:
            self.client.sock.settimeout(self._MQTT_OP_TIMEOUT_S)
        except Exception:
            pass

    def connect(self):
        """Connect to MQTT broker"""
        self.client = MQTTClient(
            self.config.MQTT_CLIENT_ID,
            self.config.MQTT_BROKER,
            port=self.config.MQTT_PORT,
            user=self.config.MQTT_USERNAME,
            password=self.config.MQTT_PASSWORD
        )

        print('Connecting to MQTT broker...')
        self.client.connect(timeout=5)
        print('MQTT connected')

        self._apply_op_timeout()

        # Publish-only client: we deliberately do NOT subscribe.
        #
        # The firmware never processes inbound commands, so subscribe()
        # buys us nothing — and it actively breaks things on the plain
        # ESP32. On a congested link the SUBACK times out, leaving the
        # SUBSCRIBE packet pinned in lwIP's TCP retransmit queue for the
        # whole session. lwIP's buffer pool is tiny here; that pinned
        # buffer starves the UDP traffic generator (ENOMEM) and, with it,
        # CSI reception — which looks exactly like a hang.

        return self.client

    def publish_state(self, current_variance, current_state, current_threshold,
                     packet_delta, dropped_delta, pps):
        """
        Publish current state to MQTT

        Args:
            current_variance: Current moving variance (or probability for ML)
            current_state: Current state (0=IDLE, 1=MOTION)
            current_threshold: Current threshold
            packet_delta: Packets processed since last publish
            dropped_delta: Packets dropped since last publish
            pps: Packets per second
        """
        state_str = 'motion' if current_state == 1 else 'idle'

        payload = {
            'movement': round(current_variance, 4),
            'threshold': round(current_threshold, 4),
            'state': state_str,
            'packets_processed': packet_delta,
            'packets_dropped': dropped_delta,
            'pps': pps,
            'timestamp': time.time()
        }

        # Blocking write bounded by a short timeout. umqtt builds the
        # PUBLISH packet from several small sock.write() calls — a
        # non-blocking socket could send a partial packet and desync the
        # MQTT stream, so we stay blocking and bound it with a timeout.
        self._apply_op_timeout()
        try:
            self.client.publish(self.base_topic, json.dumps(payload))
        except OSError as e:
            errno = e.args[0] if e.args else None
            # ENOMEM(12)/EAGAIN(11): lwIP is momentarily out of buffers and
            # the write never started — nothing partial was sent. Skip this
            # publish and keep the session; the next one will likely work.
            if errno in (11, 12):
                self._publish_skips = getattr(self, '_publish_skips', 0) + 1
                if self._publish_skips % 20 == 1:
                    print(f"MQTT publish skipped (transient ENOMEM/EAGAIN) x{self._publish_skips}")
                return
            # ETIMEDOUT / ECONNRESET / EPIPE: connection is unhealthy and the
            # stream may be desynced — propagate so the caller drops MQTT.
            print(f"Error publishing to MQTT: {e}")
            raise
        except Exception as e:
            print(f"Error publishing to MQTT: {e}")
            raise

        # Update state
        self.last_variance = current_variance
        self.last_state = current_state

    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            try:
                self.client.disconnect()
                print('MQTT disconnected')
            except Exception as e:
                print(f"Error disconnecting MQTT: {e}")

    def publish_info(self):
        """Publish a one-shot device info message to {topic}/info."""
        payload = {
            'device': self.config.MQTT_CLIENT_ID,
            'algorithm': self.detector.get_name(),
            'threshold': round(self.detector.get_threshold(), 4),
            'window_size': getattr(self.config, 'SEG_WINDOW_SIZE', None),
        }

        self._apply_op_timeout()
        try:
            self.client.publish(self.info_topic, json.dumps(payload))
        except Exception as e:
            print(f"Error publishing MQTT info: {e}")
            raise
