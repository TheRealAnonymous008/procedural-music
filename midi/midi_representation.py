import py_midicsv as pm
from midi.track import Track

from tempfile import TemporaryFile
import py_midicsv as pm


class Composition:
    def __init__(self, name: str, format = 1, division: int = 43):
        self.name = name
        self.tracks = []
        self.format = format
        self.division = division

    def add_track(self, track: int, duration: int):
        self.tracks.append(Track(track, duration))

    def get_track(self, track: int) -> Track:
        return self.tracks[track - 1]

    def finalize(self):
        header = ','.join(['0', '0', 'Header', str(self.format), str(len(self.tracks)), str(self.division)])
        eof = ','.join(['0', '0', 'End_of_file'])

        events = [header]
        for track in self.tracks:
            events.extend(track.finalize())

        events += [eof]
        return events


def make_midi(comp : Composition):
    fin = comp.finalize()
    csv_data = '\n'.join(fin)

    with TemporaryFile('w+') as output:
        output.write(csv_data)
        output.seek(0)
        midi_object = pm.csv_to_midi(output)

    with open("example_converted.mid", "wb") as output_file:
        midi_writer = pm.FileWriter(output_file)
        midi_writer.write(midi_object)


if __name__ == '__main__':
    comp = Composition("Test")
    comp.add_track(1, 1000)
    t = comp.get_track(1)

    t.add_note_on_event(0, 0, 60, 79)
    t.add_note_off_event(100, 0, 60)

    make_midi(comp)