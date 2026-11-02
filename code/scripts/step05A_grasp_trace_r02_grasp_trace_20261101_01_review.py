"""Verify and inspect already generated artifacts; no video re-decode or model/simulator."""
import json,csv,os,glob,math,xml.etree.ElementTree as ET,resource,time
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/workspace');OUT=ROOT/'artifacts/step05A_grasp_trace_review/grasp_trace_20261101_01'
assert os.getuid()!=0 and not glob.glob('/dev/nvidia*') and os.getenv('NVIDIA_VISIBLE_DEVICES')=='void'
assert not (OUT/'review_started.json').exists()
(OUT/'review_started.json').write_text(json.dumps({'time':time.time()}))
summary=json.loads((OUT/'summary.json').read_text());validation=json.loads((OUT/'trace_validation.json').read_text());scenes=json.loads((OUT/'scenes.json').read_text());correspondence=json.loads((OUT/'correspondence.json').read_text())
assert validation['pass'] and validation['rows']==25000 and len(summary['trials'])==5 and summary['decoded_frames']==14541 and len(scenes)==90 and len(correspondence)==15
for trial in range(5):
    for label in ['teacher','600','5000']:assert sum(s['trial']==trial and s['series']==label for s in scenes)==6
names=['Rotation_R','Pitch_R','Elbow_R','Wrist_Pitch_R','Wrist_Roll_R','Jaw_R']
def readcsv(p):
    with p.open() as f:return list(csv.DictReader(f))
evidence=[];counts={'trajectory_rows':0,'nearest_rows':0,'same_time_rows':0,'svg_files':0,'svg_polylines':0};ns={'s':'http://www.w3.org/2000/svg'}
for result in summary['trials']:
    trial=result['trial'];ix=f'{trial:02d}';td=readcsv(OUT/f'trial_{ix}_teacher_trajectory.csv');length=len(td)
    assert length==result['series']['teacher']['rows']
    landmarks={k:int(result['teacher_phase_starts'][k]) for k in ['approach','descend','close','lift','hold']}
    entry={'trial':trial,'landmarks':{},'late_nearest':{}}
    for label in ['teacher','600','5000']:
        rows=td if label=='teacher' else readcsv(OUT/f'trial_{ix}_{label}_trajectory.csv')
        assert len(rows)==result['series'][label]['rows'];counts['trajectory_rows']+=len(rows)
        for i,row in enumerate(rows):
            assert int(row['index'])==i and abs(float(row['post_time_s'])-float(row['pre_time_s'])-.02)<1e-8
            for n in names:
                for k in ['pre_state','action','post_state','command_delta','action_minus_prestate','ctrl_minus_poststate']:assert math.isfinite(float(row[n+'_'+k]))
        entry['landmarks'][label]={phase:{'index':i,'time_s':float(rows[i]['pre_time_s']),'state_rad':[float(rows[i][n+'_pre_state']) for n in names],'action_rad':[float(rows[i][n+'_action']) for n in names],'post_bilateral':rows[i]['post_bilateral'],'post_robot_contact':rows[i]['post_robot_contact'],'post_lift_m':float(rows[i]['post_lift_m'])} for phase,i in landmarks.items()}
        if label!='teacher':
            near=readcsv(OUT/f'trial_{ix}_{label}_nearest.csv');same=readcsv(OUT/f'trial_{ix}_{label}_same_time.csv');assert len(near)==2500 and len(same)==length
            counts['nearest_rows']+=len(near);counts['same_time_rows']+=len(same)
            selected=[]
            for i in [0,100,500,1000,1500,2000,2250,2499]:
                row=near[i];j=int(row['teacher_index']);distance=math.sqrt(sum(((float(row[n+'_act_pre'])-float(td[j][n+'_pre_state']))/summary['std_rad'][k])**2 for k,n in enumerate(names[:5])))
                assert abs(distance-float(row['distance_l2_std_arm5']))<1e-12
                assert int(row['teacher_prev_index'])==max(j-1,0) and int(row['teacher_next_index'])==min(j+1,length-1)
                selected.append({'act_index':i,'act_time':float(row['act_pre_time_s']),'teacher_index':j,'teacher_time':float(row['teacher_pre_time_s']),'distance':distance,'teacher_phase_reference_only':row['teacher_phase_reference_only'],'act_state':[float(row[n+'_act_pre']) for n in names],'teacher_state':[float(row[n+'_teacher_pre']) for n in names],'act_command':[float(row[n+'_act_command']) for n in names],'teacher_command':[float(row[n+'_teacher_command']) for n in names]})
            late=[row for row in near if float(row['act_pre_time_s'])>=40];assert len(late)==500
            entry['late_nearest'][label]={'selected_points':selected,'all_500_late_mean_abs_state_difference_rad':[sum(abs(float(row[n+'_act_pre'])-float(row[n+'_teacher_pre'])) for row in late)/len(late) for n in names],'all_500_late_mean_signed_command_difference_rad':[sum(float(row[n+'_act_command'])-float(row[n+'_teacher_command']) for row in late)/len(late) for n in names]}
    for metric,expected in [('state_action',36),('command_change',18),('tracking',36)]:
        tree=ET.parse(OUT/f'trial_{ix}_{metric}.svg');polys=tree.findall('.//s:polyline',ns);assert len(polys)==expected
        for p in polys:
            xy=[tuple(map(float,pair.split(','))) for pair in p.attrib['points'].split()];assert all(len(pt)==2 and all(math.isfinite(v) for v in pt) for pt in xy)
        counts['svg_files']+=1;counts['svg_polylines']+=len(polys)
    sheet=Image.open(OUT/f'trial_{ix}_review_sheet.png');assert sheet.size==(1920,3120)
    # Derived view of the same existing scenes, no new representative scene or decoded frame.
    focus=Image.new('RGB',(1170,1335),'white');draw=ImageDraw.Draw(focus)
    for row,source_row in enumerate([2,3,4]):
        for col,label in enumerate(['teacher','600','5000']):
            x0=col*640;y0=source_row*520+40
            crop=sheet.crop((x0+255,y0+205,x0+385,y0+340)).resize((390,405),resample=Image.Resampling.NEAREST)
            focus.paste(crop,(col*390,row*445+40));draw.text((col*390+5,row*445+10),f'{label} / source row {source_row+1}',fill='black')
    focus.save(OUT/f'trial_{ix}_focus.png');evidence.append(entry)
assert counts['trajectory_rows']==34531 and counts['nearest_rows']==25000 and counts['same_time_rows']==19062
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False))
(OUT/'artifact_validation.json').write_text(json.dumps({'pass':True,'counts':counts,'scenes':90,'decoded_frames_additional':0,'derived_focus_sheets':5,'crop_in_source_frame':[255,205,385,340],'scale':3,'method':'nearest-neighbor, existing scene rows3/4/5 only','max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
print(json.dumps({'ARTIFACT_VALIDATION_COMPLETE':True,'counts':counts}),flush=True)
