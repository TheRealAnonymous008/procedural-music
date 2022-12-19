import py_midicsv as pm

from midi.track import Track


class Composition:
    def __init__(self, name: str, format = 1, division: int = 43):
        self.name = name
        self.tracks = []
        self.format = format
        self.division = division

    def add_track(self, track: int):
        self.tracks.append(Track(int))

    def finalize(self):
        header = ','.join(['0', '0', 'Header', str(format), str(len(self.tracks)), str(self.division)])
        eof = ','.join(['0', '0', 'End_of_file'])

        events = [header]
        for track in self.tracks:
            events.extend(track.finalize())

        events.sort()
        events += [eof]
        return events


if __name__ == '__main__':
    csv_string = pm.midi_to_csv('../data/sample.midi')

    with open("example_converted.csv", "w") as f:
        f.writelines(csv_string)