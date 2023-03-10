class Track:
    def __init__(self, track, duration):
        self.track = track
        self.duration = duration
        self.midi_events: list(str)= []
        self.bpm = 120

    def finalize(self):
        start_track = str(self.track) + ", 0, Start_track"
        end_track = str(self.track) + ", " + str(self.duration) + ", End_track, "

        events = [start_track] + self.midi_events + [end_track]
        return events

    def add_note_on_event(self, time, channel, note, velocity):
        args = [str(self.track), str(time), "Note_on_c", str(channel), str(note), str(velocity)]
        self.midi_events.append(','.join(args))

    def add_note_off_event(self, time, channel, note, velocity=0):
        args = [str(self.track), str(time), "Note_off_c", str(channel), str(note), str(velocity)]
        self.midi_events.append(','.join(args))

    def add_pitch_bend_event(self, time, channel, value):
        args = [str(self.track), str(time), "Pitch_bend_c", str(channel), str(value)]
        self.midi_events.append(','.join(args))

    def add_control_event(self, time, channel, control_num, value):
        args = [str(self.track), str(time), "Control_c", str(channel), str(control_num), str(value)]
        self.midi_events.append(','.join(args))

    def add_program_event(self, time, channel, program_num):
        args = [str(self.track), str(time), "Program_c", str(channel), str(program_num)]
        self.midi_events.append(','.join(args))

    def add_channel_aftertouch_event(self, time, channel, value):
        args = [str(self.track), str(time), "Channel_aftertouch_c", str(channel), str(value)]
        self.midi_events.append(','.join(args))

    def add_poly_aftertouch_event(self, time, channel, note, value):
        args = [str(self.track), str(time), "Poly_aftertouch_c", str(channel), str(note), str(value)]
        self.midi_events.append(','.join(args))

    def add_tempo_event(self, time, bpm):
        self.bpm = bpm
        args = [str(self.track), str(time), "Tempo", str(60000000 // bpm)]
        self.midi_events.append(','.join(args))