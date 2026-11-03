"""Display crops at native pixel scale; no resampling of the selected region."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];out=root/'output/staged_reconstruction'/a.stage
    result=Image.new('RGB',(2250,1530))
    for column,view in enumerate(['fixed86','fixed188','original']):
        for row,number in enumerate([2,86,188]):
            image=Image.open(out/f'{view}_t{number:04d}_native.png');assert image.size==(1600,900)
            crop=image.crop((450,160,1200,650));x=column*750;y=row*510
            result.paste(crop,(x,y+20));ImageDraw.Draw(result).text((x+4,y+4),f'{a.stage} {view} t{number}; native crop x450:1200 y160:650',fill='white')
    result.save(out/'native_crop_review.png')


if __name__=='__main__':main()
