"""Run with python forecast_gui.py from the project folder. CPU inference on Windows/Linux."""
from pathlib import Path
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, font
import pandas as pd
import numpy as np
from forecastprep.core import load_config, stations
from forecastprep.desktop import load_bundle, fetch_inputs, forecast, inspect_inputs, validate_live_config

ROOT=Path(__file__).resolve().parent
BG,WHITE,INK,TEAL,NAVY,SOFT='#eff3f5','#ffffff','#173544','#006c63','#143844','#4d646d'


class ForecastWindow:
    def __init__(self, root):
        self.root=root;root.title('Weather forecasts | HRRR + GOES')
        root.geometry('1160x850');root.minsize(1050,700);root.configure(bg=BG)
        available={f.lower():f for f in font.families(root)}
        self.family=next((available[f] for f in ['segoe ui','dejavu sans','liberation sans','arial'] if f in available),font.nametofont('TkDefaultFont').actual('family'))
        self.config=load_config(ROOT/'configs/live.yaml')
        self.locations=stations(self.config['stations'])
        self.bundle=None;self.snapshot=None;self.overrides={};self.answers=[];self.image_array=None
        self.busy=False;self.revision=0;self.messages=queue.Queue();self.controls=[]
        self.station=tk.StringVar(value=self.locations[0]['station_id'])
        self.budget=tk.StringVar(value='3.15');self.strategy=tk.StringVar(value='learned')
        self.status=tk.StringVar(value='Choose a location. Load a trained model after ARC training, or inspect the included example now.')
        self.source=tk.StringVar(value='No forecast inputs loaded')
        self.model_label=tk.StringVar(value='No trained model loaded')
        self.config_label=tk.StringVar(value='Configuration: live.yaml | GOES-19')
        self.place=tk.StringVar();self.leads={h:tk.BooleanVar(value=True) for h in [6,12,24,48]}
        self._style();self._layout();self._location_changed()
        for var in [self.budget,self.strategy,*self.leads.values()]:var.trace_add('write',lambda *_:self.invalidate())
        root.after(100,self._poll)

    def _style(self):
        s=ttk.Style(self.root);s.theme_use('clam')
        s.configure('.',font=(self.family,10),background=WHITE,foreground=INK)
        s.configure('TButton',padding=(10,7));s.configure('TCombobox',padding=5)
        s.configure('Action.TButton',background=TEAL,foreground=WHITE,font=(self.family,11,'bold'))
        s.map('Action.TButton',background=[('active','#005d54')])
        s.configure('Treeview',rowheight=29,fieldbackground=WHITE)
        s.configure('Treeview.Heading',font=(self.family,10,'bold'))

    def button(self,parent,text,fn,**kw):
        b=ttk.Button(parent,text=text,command=fn,**kw);self.controls.append((b,'normal'));return b

    def label(self,parent,text=None,variable=None,size=10,color=INK,bg=WHITE):
        return tk.Label(parent,text=text,textvariable=variable,font=(self.family,size),fg=color,bg=bg,anchor='w',justify='left',wraplength=1060)

    def _layout(self):
        top=tk.Frame(self.root,bg=NAVY,padx=22,pady=17);top.pack(fill='x')
        self.label(top,'Weather forecasts',size=23,color=WHITE,bg=NAVY).pack(side='left')
        self.label(top,'HRRR + GOES  /  Temperature °C',color='#b6d9d4',bg=NAVY).pack(side='right')
        body=tk.Frame(self.root,bg=BG,padx=15,pady=12);body.pack(fill='both',expand=True);body.columnconfigure(0,weight=1);body.rowconfigure(4,weight=1)
        toolbar=tk.Frame(body,bg=WHITE,padx=12,pady=10);toolbar.grid(row=0,column=0,sticky='ew')
        for text,fn in [('Load model...',self.open_model),('Load config...',self.open_config),
                        ('Load stations...',self.open_stations),('Open saved inputs...',self.open_snapshot),('View example',self.example)]:
            self.button(toolbar,text,fn).pack(side='left',padx=(0,7))
        info=tk.Frame(body,bg=WHITE,padx=12,pady=6);info.grid(row=1,column=0,sticky='ew')
        self.label(info,variable=self.model_label,color=SOFT).pack(anchor='w')
        self.label(info,variable=self.config_label,color=SOFT).pack(anchor='w')
        row=tk.Frame(body,bg=WHITE,padx=12,pady=12);row.grid(row=2,column=0,sticky='ew',pady=(10,0))
        self.label(row,'Location').grid(row=0,column=0,sticky='w')
        self.station_box=ttk.Combobox(row,textvariable=self.station,values=[s['station_id'] for s in self.locations],state='readonly',width=14)
        self.station_box.grid(row=1,column=0,sticky='w',padx=(0,14));self.controls.append((self.station_box,'readonly'))
        self.station_box.bind('<<ComboboxSelected>>',self._location_changed)
        self.label(row,variable=self.place).grid(row=1,column=1,sticky='w',padx=8)
        self.button(row,'Load latest forecast inputs',self.latest,style='Action.TButton').grid(row=1,column=2,padx=10)
        self.label(row,'Inputs follow the latest eligible HRRR cycle. All times below are UTC.',color=SOFT).grid(row=2,column=0,columnspan=3,sticky='w',pady=(9,0))
        self.label(body,variable=self.source,bg=BG,color=SOFT).grid(row=3,column=0,sticky='ew',pady=9)
        tabs=ttk.Notebook(body);tabs.grid(row=4,column=0,sticky='nsew')
        input_tab=ttk.Frame(tabs,padding=12);tabs.add(input_tab,text='Numerical inputs')
        self.label(input_tab,'Double-click a value to edit it. Empty values are missing observations.',color=SOFT).pack(anchor='w',pady=(0,8))
        self.inputs=ttk.Treeview(input_tab,columns=('lead','feature','value'),show='headings',height=4)
        for name,width in [('lead',90),('feature',260),('value',160)]:
            self.inputs.heading(name,text=name.title());self.inputs.column(name,width=width)
        scroll=ttk.Scrollbar(input_tab,orient='vertical',command=self.inputs.yview)
        self.inputs.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');self.inputs.pack(fill='both',expand=True)
        self.inputs.bind('<Double-1>',self.edit_input);self.feature_keys={}
        image_tab=ttk.Frame(tabs,padding=12);tabs.add(image_tab,text='Satellite overview')
        self.image_tab=image_tab;image_tab.bind('<Configure>',self.resize_preview)
        self.preview=tk.Label(image_tab,bg=WHITE,text='Load inputs to see the C13 satellite overview.');self.preview.pack(side='left',padx=12)
        self.preview.bind('<Button-1>',self.large_image)
        self.image_note=self.label(image_tab,'');self.image_note.pack(side='left',anchor='n',padx=12)
        options=tk.Frame(body,bg=WHITE,padx=12,pady=9);options.grid(row=5,column=0,sticky='ew',pady=(10,0))
        self.label(options,'Leads from HRRR initialization:').pack(side='left')
        for h,var in self.leads.items():
            b=ttk.Checkbutton(options,text=f'{h} h',variable=var);b.pack(side='left',padx=5);self.controls.append((b,'normal'))
        self.label(options,'Policy budget:').pack(side='left',padx=(15,4))
        ent=ttk.Entry(options,textvariable=self.budget,width=6);ent.pack(side='left');self.controls.append((ent,'normal'))
        combo=ttk.Combobox(options,textvariable=self.strategy,values=['learned','static','random'],state='readonly',width=9)
        combo.pack(side='left',padx=8);self.controls.append((combo,'readonly'))
        self.button(options,'Calculate forecast',self.calculate,style='Action.TButton').pack(side='right')
        self.results=ttk.Treeview(body,columns=('lead','valid','pred','raw','delta','cost'),show='headings',height=4)
        for col,label,width in [('lead','Lead',65),('valid','Valid time (UTC)',225),('pred','Prediction °C',120),('raw','Raw HRRR °C',120),('delta','MOS adjustment °C',150),('cost','Optional cost',100)]:
            self.results.heading(col,text=label);self.results.column(col,width=width,anchor='center')
        self.results.grid(row=6,column=0,sticky='ew',pady=10)
        footer=tk.Frame(body,bg=BG);footer.grid(row=7,column=0,sticky='ew')
        status_label=self.label(footer,variable=self.status,bg=BG,color=SOFT);status_label.configure(wraplength=850);status_label.pack(side='left',fill='x',expand=True)
        self.button(footer,'Save results...',self.save_results).pack(side='right')

    def selected_station(self):
        return next(s for s in self.locations if s['station_id']==self.station.get())

    def invalidate(self):
        self.revision+=1;self.answers=[]
        if hasattr(self,'results'):self.results.delete(*self.results.get_children())

    def _location_changed(self,event=None):
        self.invalidate();self.overrides={}
        s=self.selected_station();self.place.set(f"{s['latitude']:.4f}°, {s['longitude']:.4f}°  |  {s['iem_network']}")
        self.render_inputs()

    def _set_locations(self,locations):
        self.locations=locations;ids=[s['station_id'] for s in locations]
        if not ids:raise ValueError('Station list is empty')
        self.station_box.configure(values=ids)
        if self.station.get() not in ids:self.station.set(ids[0])
        self._location_changed()

    def open_config(self):
        path=filedialog.askopenfilename(filetypes=[('YAML config','*.yaml *.yml')])
        if not path:return
        try:
            c=load_config(path);locations=stations(c['stations'])
            if self.bundle:validate_live_config(c,self.bundle)
            self.config=c;self.snapshot=None;self._set_locations(locations)
            for lead,var in self.leads.items():var.set(lead in c['leads'])
            self.config_label.set(f"Configuration: {Path(path).name} | GOES-{c['satellite']['number']}")
        except Exception as e:messagebox.showerror('Configuration',str(e))

    def open_stations(self):
        path=filedialog.askopenfilename(filetypes=[('Station CSV','*.csv')])
        if not path:return
        try:self._set_locations(stations(path))
        except Exception as e:messagebox.showerror('Station list',str(e))

    def open_model(self):
        path=filedialog.askopenfilename(filetypes=[('Exported PyTorch model','*.pt')])
        if not path:return
        try:
            bundle=load_bundle(path);validate_live_config(self.config,bundle)
            self.bundle=bundle;self.invalidate()
            c=bundle['config'];self.model_label.set(f"{Path(path).name} | {c['mode'].upper()} | {c['kind']} | seed {c['seed']}")
            self.status.set('Model loaded. Load inputs for your chosen location, then calculate. Budget applies to masked models.')
        except Exception as e:messagebox.showerror('Model',str(e))

    def open_snapshot(self):
        path=filedialog.askdirectory()
        if path:self.use_snapshot(Path(path))

    def example(self):
        self.use_snapshot(ROOT/'example_data/source_check_2021')
        self.status.set('Historical 2021 example for inspection; its smaller images do not match the production model.')

    def use_snapshot(self,path):
        try:
            frame=inspect_inputs(path)
            if frame.init_utc.nunique()!=1:raise ValueError('Open a single-cycle input folder, not the full training archive.')
            self.snapshot=path;self.overrides={};self.invalidate();self.render_inputs()
        except Exception as e:messagebox.showerror('Inputs',str(e))

    def render_inputs(self):
        self.inputs.delete(*self.inputs.get_children());self.feature_keys={}
        self.preview.configure(image='',text='No satellite image loaded');self.preview.image=None
        self.image_array=None;self.image_note.configure(text='')
        if self.snapshot is None:self.source.set('No forecast inputs loaded');return
        frame=inspect_inputs(self.snapshot);frame=frame[frame.station_id==self.station.get()]
        if frame.empty:self.source.set('This snapshot has no inputs for the selected station. Load its latest inputs.');return
        first=frame.iloc[0]
        self.source.set(f"{first.station_id} | Cycle {first.init_utc} | Observation {first.input_observation_utc}")
        cols=['obs_current_c','obs_age_minutes','obs_lag3_c','obs_lag6_c','obs_lag24_c','hrrr_t2m_c','hrrr_u10_ms','hrrr_v10_ms','hrrr_d2m_c']
        for row in frame.itertuples():
            for name in cols:
                if name not in frame:continue
                value=self.overrides.get(row.sample_id,{}).get(name,getattr(row,name))
                iid=self.inputs.insert('', 'end',values=(f'{row.lead_h} h',name,f'{value:.3f}' if np.isfinite(value) else 'missing'))
                self.feature_keys[iid]=(row.sample_id,name)
        if first.overview_path:
            try:
                manifest=json.loads((self.snapshot/'manifest.json').read_text())
                meta=manifest['samples'][first.sample_id]['satellite'].get('13',{})
                image_path=meta.get('path',first.overview_path)
                with np.load(self.snapshot/image_path,allow_pickle=False) as data:arr=data['image_k'][0]
                self.image_array=np.uint8(np.clip((310-np.nan_to_num(arr,nan=310))/110,0,1)*255)
                self.resize_preview()
                self.image_note.configure(text='C13 brightness temperature\nBright = colder cloud tops\nDark = warmer surface/clouds\nClick the image to enlarge.\n\n'+str(meta.get('scan_end_utc',meta.get('end_utc','')))[:80])
            except Exception as e:self.preview.configure(text='Preview unavailable: '+str(e))

    def resize_preview(self,event=None):
        if self.image_array is None:return
        from PIL import Image,ImageTk
        side=max(40,min(240,self.image_tab.winfo_height()-28))
        im=ImageTk.PhotoImage(Image.fromarray(self.image_array).resize((side,side)))
        self.preview.configure(image=im,text='',cursor='hand2');self.preview.image=im

    def large_image(self,event=None):
        if self.image_array is None:return
        from PIL import Image,ImageTk
        window=tk.Toplevel(self.root);window.title('C13 satellite overview | '+self.station.get())
        window.configure(bg=WHITE)
        self.label(window,self.source.get()).pack(padx=15,pady=12)
        im=ImageTk.PhotoImage(Image.fromarray(self.image_array).resize((640,640)))
        label=tk.Label(window,image=im,bg=WHITE);label.image=im;label.pack(padx=15,pady=(0,15))

    def edit_input(self,event):
        if self.busy:return
        iid=self.inputs.identify_row(event.y)
        if iid not in self.feature_keys:return
        sid,name=self.feature_keys[iid]
        value=simpledialog.askfloat('Edit input',name+' (C, minutes, or m/s as named)',parent=self.root)
        if value is None:return
        if not np.isfinite(value) or (name.endswith('_c') and not -100<=value<=65) or (name=='obs_age_minutes' and value<0):
            messagebox.showerror('Input','Enter a finite value in the physical range.');return
        self.overrides.setdefault(sid,{})[name]=value;self.invalidate();self.render_inputs()
        self.status.set('Inputs edited manually. Recalculate to update the forecast.')

    def start(self,work,done,text):
        if self.busy:return
        self.busy=True;token=self.revision;self.status.set(text)
        for widget,_ in self.controls:widget.configure(state='disabled')
        def runner():
            try:self.messages.put((token,done,work(),None))
            except Exception as e:self.messages.put((token,done,None,str(e)))
        threading.Thread(target=runner,daemon=True).start()

    def _poll(self):
        try:
            token,done,value,error=self.messages.get_nowait()
            self.busy=False
            for widget,state in self.controls:widget.configure(state=state)
            if token==self.revision:
                if error:self.status.set('Action failed; see details.');messagebox.showerror('Forecast',error)
                else:done(value)
        except queue.Empty:pass
        self.root.after(100,self._poll)

    def latest(self):
        self.snapshot=None;self.overrides={};self.invalidate();self.render_inputs()
        cfg=self.config.copy();station=self.selected_station().copy()
        def done(path):
            self.use_snapshot(path)
            self.status.set('Inputs loaded. Check the cycle time; short leads may already be in the past. Satellite/model transfer still requires validation.')
        self.start(lambda:fetch_inputs(cfg,station,ROOT/'data/gui_cache'),done,'Downloading one forecast cycle and satellite inputs; this may take several minutes...')

    def calculate(self):
        if self.bundle is None:messagebox.showinfo('Model needed','Load an exported model from ARC after training. You can inspect current inputs before training.');return
        if self.snapshot is None:messagebox.showinfo('Inputs needed','Load forecast inputs for the chosen station first.');return
        try:budget=float(self.budget.get())
        except ValueError:messagebox.showerror('Budget','Enter a numeric budget.');return
        leads=[h for h,var in self.leads.items() if var.get()]
        sid=self.station.get();strategy=self.strategy.get();self.invalidate()
        def done(answers):
            self.answers=answers
            for a in answers:
                delta='--' if a['correction_c'] is None else f"{a['correction_c']:+.2f}"
                self.results.insert('', 'end',values=(f"{a['lead_h']} h",a['valid_utc'],f"{a['prediction_c']:.2f}",f"{a['hrrr_t2m_c']:.2f}",delta,f"{a['optional_cost']:.2f}"))
            heldout=set((self.bundle['schema']['config'].get('splits') or {}).get('heldout_stations',[]))
            trained={s['station_id'] for s in self.bundle['stations']}-heldout
            self.status.set('Forecast calculated on CPU.'+(' Manually edited inputs.' if self.overrides else '')+(' Location outside the training station list; accuracy unvalidated.' if sid not in trained else ''))
        self.start(lambda:forecast(self.bundle,self.snapshot,leads,self.overrides,budget,strategy,sid),done,'Calculating temperature forecasts...')

    def save_results(self):
        if not self.answers:messagebox.showinfo('Results','Calculate a forecast first.');return
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')])
        if path:pd.DataFrame(self.answers).to_csv(path,index=False);self.status.set('Forecast results saved.')


def main():
    import torch
    torch.set_num_threads(4)
    root=tk.Tk();ForecastWindow(root);root.mainloop()


if __name__=='__main__':main()
