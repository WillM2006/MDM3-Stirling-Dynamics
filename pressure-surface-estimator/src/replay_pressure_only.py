"""Replay a CSV or stdin pressure stream without reading any future rows."""
import argparse,csv,sys
from pressure_only_live import Calibration,PressureOnlyLive

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--pressures',required=True,help='CSV path, or - for stdin')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    live=PressureOnlyLive(Calibration.load(args.calibration))
    source=sys.stdin if args.pressures=='-' else open(args.pressures,newline='')
    with source,open(args.output,'w',newline='') as output:
        reader=csv.reader(source);header=next(reader)
        if len(header)!=7 or header[0]!='time_s':raise ValueError('CSV must contain time_s and six pressure columns only')
        writer=csv.writer(output)
        writer.writerow(['time_s']+[f'eta_x{x:g}_y{y:g}_m' for x,y in live.calibration.targets])
        for row in reader:
            if len(row)!=7:raise ValueError('Exactly time and six pressures required')
            timestamp,*pressure=map(float,row)
            surface=live.push(timestamp,pressure)
            if surface is not None:writer.writerow([timestamp,*surface])

if __name__=='__main__':main()
