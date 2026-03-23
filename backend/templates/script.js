
/* ================================
   NAVIGATION
================================ */
const navLinks = document.querySelectorAll('nav ul li a');
const sections = ["home","dashboard","players","about"];

function navigateTo(sectionId){
  sections.forEach(id=>{
    const el=document.getElementById(id);
    if(el) el.style.display=(id===sectionId)?"block":"none";
  });

  navLinks.forEach(link=>{
    link.classList.toggle(
      "active",
      link.getAttribute("href")==="#"+sectionId
    );
  });
}

navLinks.forEach(link=>{
  link.addEventListener("click",e=>{
    e.preventDefault();
    navigateTo(link.getAttribute("href").substring(1));
  });
});

navigateTo("home");


/* ================================
   DATA
================================ */
const TEAMS=[
"Chennai Super Kings",
"Mumbai Indians",
"Royal Challengers Bangalore",
"Kolkata Knight Riders",
"Rajasthan Royals",
"Sunrisers Hyderabad",
"Delhi Capitals",
"Gujarat Titans",
"Lucknow Super Giants",
"Punjab Kings"
];


/* ================================
   INIT
================================ */
document.addEventListener("DOMContentLoaded",()=>{
 fillDropdown("team1");
 fillDropdown("team2");
 fillDropdown("batting-team");
 fillDropdown("bowling-team");
 setupMatchSync();
});


function fillDropdown(id){
 const el=document.getElementById(id);
 if(!el) return;

 TEAMS.forEach(t=>{
   const opt=document.createElement("option");
   opt.value=t;
   opt.textContent=t;
   el.appendChild(opt);
 });
}


/* ================================
   TEAM SYNC + VALIDATION
================================ */
function setupMatchSync(){

const team1=document.getElementById("team1");
const team2=document.getElementById("team2");
const toss=document.getElementById("toss-winner");

function update(){

 const t1=team1.value;
 const t2=team2.value;

 /* prevent same team */
 [...team2.options].forEach(o=>o.disabled=o.value===t1);
 [...team1.options].forEach(o=>o.disabled=o.value===t2);

 /* toss dropdown */
 toss.innerHTML="";
 [t1,t2].forEach(t=>{
   if(!t) return;
   const o=document.createElement("option");
   o.value=t;
   o.textContent=t;
   toss.appendChild(o);
 });
}

team1.addEventListener("change",update);
team2.addEventListener("change",update);
}


/* ================================
   MATCH PREDICT
================================ */
window.predictWinner=async function(){

const result=document.getElementById("winnerResult");

const t1=document.getElementById("team1").value;
const t2=document.getElementById("team2").value;
const toss=document.getElementById("toss-winner").value;
const decision=document.getElementById("toss-decision").value;
const venue=document.getElementById("venue").value;

if(!t1||!t2){
 result.innerHTML="Select teams";
 return;
}
if(t1===t2){
 result.innerHTML="Teams must differ";
 return;
}
if(!toss){
 result.innerHTML="Select toss winner";
 return;
}

result.innerHTML="Predicting...";

try{

const res=await fetch("/predict",{
 method:"POST",
 headers:{"Content-Type":"application/json"},
 body:JSON.stringify({
  team1:t1,
  team2:t2,
  toss_winner:toss,
  toss_decision:decision,
  venue:venue
 })
});

const data=await res.json();

result.innerHTML=`
<div class="glass-card">
<div class="pred-title">PREDICTED WINNER</div>
<div class="pred-winner">${data.predicted_winner}</div>
  <div class="teams-row">
    <div class="team-box left">
      <div class="team-name">${t1}</div>
      <div class="team-prob">${(data.team1_prob*100).toFixed(2)}%</div>
    </div>

    <div class="team-box right">
      <div class="team-name">${t2}</div>
      <div class="team-prob">${(data.team2_prob*100).toFixed(2)}%</div>
    </div>
  </div>

  <div class="conf-bar">
    <div class="conf-fill" id="confFill"></div>
  </div>
<div style="margin-top:8px">
Confidence ${(data.confidence*100).toFixed(2)}%
</div>
</div>
`;

setTimeout(()=>{
 document.getElementById("confFill").style.width=
 (data.confidence*100)+"%";
},100);

}catch(e){
 result.innerHTML="Server error";
 console.log(e);
}
};
/* ===========================
search box
==============================*/

  /* ===============================
    TEAM INFO DROPDOWN FILL
  ================================*/
function fillTeamInfoDropdown(){
    const select = document.getElementById("team-info-select");
    if(!select) return;

    TEAMS.forEach(team=>{
      const opt = document.createElement("option");
      opt.value = team;
      opt.textContent = team;
      select.appendChild(opt);
    });

    select.addEventListener("change", getTeamInfo);
  }

  document.addEventListener("DOMContentLoaded", fillTeamInfoDropdown);


  /* ===============================
    TEAM NAME → CODE MAP
  ================================*/
  const TEAM_CODE = {
  "Chennai Super Kings":"CSK",
  "Mumbai Indians":"MI",
  "Royal Challengers Bangalore":"RCB",
  "Kolkata Knight Riders":"KKR",
  "Rajasthan Royals":"RR",
  "Sunrisers Hyderabad":"SRH",
  "Delhi Capitals":"DC",
  "Gujarat Titans":"GT",
  "Lucknow Super Giants":"LSG",
  "Punjab Kings":"PBKS"
  };


  /* ===============================
    FETCH TEAM INFO
  ================================*/
  async function getTeamInfo(){

  const teamName = document.getElementById("team-info-select").value;
  if(!teamName) return;

  const code = TEAM_CODE[teamName];

  try{

  const res = await fetch("/team-info/"+code);
  const data = await res.json();

  if(data.error){
  alert("Team info not found");
  return;
  }

  const info = data.info;

  document.getElementById("infoTitles").textContent = info.titles || "-";
  document.getElementById("infoCaptain").textContent = info.captain || "-";
  document.getElementById("infoCoach").textContent = info.coach || "-";
  document.getElementById("infoHome").textContent = info.home || "-";
  document.getElementById("infoFounded").textContent = info.founded || "-";

  document.getElementById("teamInfoCard").style.display="block";

  }catch(e){
  console.log(e);
  alert("Server error");
  }

  }

/*==================================
   players analysis
=================================*/
async function searchPlayer(){

const name = document.getElementById("playerInput").value;
const card = document.getElementById("playerCard");

if(!name){
 card.innerHTML="Enter player name";
 return;
}

card.innerHTML="Searching...";

try{

const res = await fetch("/player-search/"+name);
const data = await res.json();

if(data.error){
 card.innerHTML="Player not found";
 return;
}

card.innerHTML = `
<div class="player-card">

<h2>${data.name}</h2>

<div class="stats-row">

<div class="stat-box">
Runs<br>
<b>${data.runs}</b>
</div>

<div class="stat-box">
Wickets<br>
<b>${data.wickets}</b>
</div>

<div class="stat-box">
Matches<br>
<b>${data.matches}</b>
</div>

</div>
</div>
`;

}catch(e){
 card.innerHTML="Server error";
}
}
const input = document.getElementById("playerInput");
const suggestions = document.getElementById("suggestions");
const card = document.getElementById("playerCard");

input.addEventListener("input", async () => {

  const q = input.value.trim();
  if(q.length < 2){
    suggestions.innerHTML="";
    return;
  }

  const res = await fetch(`/api/player-search?q=${q}`);
  const data = await res.json();

  suggestions.innerHTML="";

  data.forEach(p=>{
    const div = document.createElement("div");
    div.textContent = p.name;
    div.onclick = ()=>showPlayer(p);
    suggestions.appendChild(div);
  });

});

function showPlayer(p){

  suggestions.innerHTML="";

  card.innerHTML = `
  <div class="player-card">
    <h2>${p.name}</h2>

    <div style="display:flex;justify-content:space-around;margin-top:20px">
      <div>
        <div style="color:#03a9f4">RUNS</div>
        <div style="font-size:22px">${p.runs}</div>
      </div>

      <div>
        <div style="color:#03a9f4">WICKETS</div>
        <div style="font-size:22px">${p.wickets}</div>
      </div>

      <div>
        <div style="color:#03a9f4">MATCHES</div>
        <div style="font-size:22px">${p.matches}</div>
      </div>
    </div>
  </div>
  `;
}


/* ===================================
   STATIC IPL WINNERS DATA
=================================== */

const IPL_DATA = [
{year:2025, winner:"Royal Challengers Bengaluru", runner:"Punjab Kings", result:"Won by 6 runs"},
{year:2024, winner:"Kolkata Knight Riders", runner:"Sunrisers Hyderabad", result:"Won by 8 wickets"},
{year:2023, winner:"Chennai Super Kings", runner:"Gujarat Titans", result:"Won by 5 wickets"},
{year:2022, winner:"Gujarat Titans", runner:"Rajasthan Royals", result:"Won by 7 wickets"},
{year:2021, winner:"Chennai Super Kings", runner:"Kolkata Knight Riders", result:"Won by 27 runs"},
{year:2020, winner:"Mumbai Indians", runner:"Delhi Capitals", result:"Won by 5 wickets"},
{year:2019, winner:"Mumbai Indians", runner:"Chennai Super Kings", result:"Won by 1 run"},
{year:2018, winner:"Chennai Super Kings", runner:"Sunrisers Hyderabad", result:"Won by 8 wickets"},
{year:2017, winner:"Mumbai Indians", runner:"Rising Pune Supergiant", result:"Won by 1 run"},
{year:2016, winner:"Sunrisers Hyderabad", runner:"Royal Challengers Bangalore", result:"Won by 8 runs"},
{year:2015, winner:"Mumbai Indians", runner:"Chennai Super Kings", result:"Won by 41 runs"},
{year:2014, winner:"Kolkata Knight Riders", runner:"Kings XI Punjab", result:"Won by 3 wickets"},
{year:2013, winner:"Mumbai Indians", runner:"Chennai Super Kings", result:"Won by 23 runs"},
{year:2012, winner:"Kolkata Knight Riders", runner:"Chennai Super Kings", result:"Won by 5 wickets"},
{year:2011, winner:"Chennai Super Kings", runner:"Royal Challengers Bangalore", result:"Won by 58 runs"},
{year:2010, winner:"Chennai Super Kings", runner:"Mumbai Indians", result:"Won by 22 runs"},
{year:2009, winner:"Deccan Chargers", runner:"Royal Challengers Bangalore", result:"Won by 6 runs"},
{year:2008, winner:"Rajasthan Royals", runner:"Chennai Super Kings", result:"Won by 3 wickets"},
];


/* ===================================
   RENDER CARDS
=================================== */

function renderWinners(){

  const grid = document.getElementById("winnersGrid");
  if(!grid) return;

  // newest first
  IPL_DATA.sort((a,b)=>b.year-a.year);

  IPL_DATA.forEach(item=>{
    const card = document.createElement("div");
    card.className = "winner-card";

    card.innerHTML = `
      <div class="winner-year">IPL ${item.year}</div>
      <div class="winner-team">${item.winner}</div>
      <div class="runner-team">vs ${item.runner}</div>
      <div class="result-text">${item.result}</div>
    `;

    grid.appendChild(card);
  });
}

document.addEventListener("DOMContentLoaded", renderWinners);

/* ================================
   SCORE PREDICT
================================ */
window.predictScore=async function(){

const result=document.getElementById("score-predictor-result");

const batting=document.getElementById("batting-team").value;
const bowling=document.getElementById("bowling-team").value;
const venue=document.getElementById("score_venue").value;
const overs=document.getElementById("overs").value;
const runs=document.getElementById("runs").value;
const wickets=document.getElementById("wickets").value;
const last5=document.getElementById("runs-last-5").value;

if(!batting||!bowling){
 result.innerHTML="Select teams";
 return;
}
if(batting===bowling){
 result.innerHTML="Teams must differ";
 return;
}

result.innerHTML="Predicting...";

try{

const res=await fetch("/predict-score",{
 method:"POST",
 headers:{"Content-Type":"application/json"},
 body:JSON.stringify({
  batting_team:batting,
  bowling_team:bowling,
  venue:venue,
  overs:overs,
  runs:runs,
  wickets:wickets,
  last5:last5
 })
});

const data=await res.json();

result.innerHTML=`
<div class="score-card">
<div style="font-size:12px;color:">
PREDICTED SCORE
</div>

<div class="score-big">${data.predicted_score}</div>

<div class="score-meta">
<div>Run Rate<br><b>${(runs/overs).toFixed(2)}</b></div>
<div>Wickets Left<br><b>${10-wickets}</b></div>
</div>
</div>
`;
const score = data.predicted_score;

const min = score - 15;
const max = score + 15;

result.innerHTML = `
<div class="score-card">

<div style="color:#03a9f4;font-size:22px;letter-spacing:2px;padding-bottom:10px">
PREDICTED FINAL SCORE
</div>

<div class="score-big">${score}</div>

<div style="opacity:.7;margin-bottom:10px">
Range: ${min} - ${max}
</div>

<div class="score-meta">
<div>
Run Rate<br>
<b>${(runs/overs).toFixed(2)}</b>
</div>

<div>
Wickets Remaining<br>
<b>${10 - wickets}</b>
</div>
</div>

</div>
`;


}catch(e){
 result.innerHTML="Server error";
 console.log(e);
}
};
/*======================
player search
=================*/
document.addEventListener("DOMContentLoaded", () => {

const input = document.getElementById("playerInput");
const suggestions = document.getElementById("suggestions");
const card = document.getElementById("playerCard");

let timer=null;

input.addEventListener("input",()=>{

clearTimeout(timer);
const q=input.value.trim();

if(q.length<2){
 suggestions.innerHTML="";
 return;
}

timer=setTimeout(async()=>{

const res=await fetch(`/api/suggest?q=${q}`);
const data=await res.json();

suggestions.innerHTML="";

data.forEach(name=>{
 const div=document.createElement("div");
 div.textContent=name;
 div.className="suggest-item";

 div.onclick=()=>loadPlayer(name);

 suggestions.appendChild(div);
});

},250);

});


async function loadPlayer(name){

suggestions.innerHTML="";
input.value=name;

card.innerHTML="Loading...";

const res=await fetch(`/api/player?name=${name}`);
const d=await res.json();

const initials=name.split(" ").map(w=>w[0]).join("").slice(0,2);

card.innerHTML=`
<div class="player-card">

<div class="avatar">${initials}</div>

<div class="player-info">
<h3>${d.name}</h3>

<div class="stats-grid">
<div><span>Runs</span><br><b>${d.runs}</b></div>
<div><span>Matches</span><br><b>${d.matches}</b></div>
<div><span>SR</span><br><b>${d.strike_rate}</b></div>
<div><span>4s</span><br><b>${d.fours}</b></div>
<div><span>6s</span><br><b>${d.sixes}</b></div>
<div><span>Wkts</span><br><b>${d.wickets}</b></div>
<div><span>Econ</span><br><b>${d.economy}</b></div>
</div>

<div class="chart-row">

<div class="chart-box">
<div class="chart-title">Batting Impact</div>
<div class="bar">
<div class="bar-fill" id="batBar"></div>
</div>
</div>

<div class="chart-box">
<div class="chart-title">Bowling Impact</div>
<div class="bar">
<div class="bar-fill" id="bowlBar"></div>
</div>
</div>

</div>

</div>
</div>
`;

setTimeout(()=>{
 document.getElementById("batBar").style.width=
 Math.min(d.runs/6000*100,100)+"%";

 document.getElementById("bowlBar").style.width=
 Math.min(d.wickets/200*100,100)+"%";
},100);

}

});

/* ================================
   PLAYER ANALYTICS
================================ */
async function loadTop(){

  const team = document.getElementById("teamCode").value;
  const result = document.getElementById("playerResult");

  result.style.display = "block";
  result.innerHTML = "Loading...";

  try{
    const res = await fetch(`/players/top/${team}`);
    const data = await res.json();

    result.innerHTML =
      "<b>TOP BATTERS</b>\n\n" +
      formatList(data.top_batters) +
      "\n\n<b>TOP BOWLERS</b>\n\n" +
      formatList(data.top_bowlers);

  }catch(err){
    result.innerHTML = "Server error";
  }
}


async function loadSeason(){

  const team = document.getElementById("teamCode").value;
  const year = document.getElementById("seasonYear").value;
  const result = document.getElementById("playerResult");

  if(!year){
    alert("Enter season year");
    return;
  }

  result.style.display = "block";
  result.innerHTML = "Loading season data...";

  try{
    const res = await fetch(`/players/top-season/${team}/${year}`);
    const data = await res.json();

    if(data.message){
      result.innerHTML = data.message;
      return;
    }

    result.innerHTML =
      "<b>SEASON " + year + "</b>\n\n" +
      "<b>TOP BATTERS</b>\n\n" +
      formatList(data.top_batters) +
      "\n\n<b>TOP BOWLERS</b>\n\n" +
      formatList(data.top_bowlers);

  }catch(err){
    result.innerHTML = "Server error";
  }
}


function formatList(obj){
  let text = "";
  for(let name in obj){
    text += name + " : " + obj[name] + "\n";
  }
  if(text==="") text="No data";
  return text;
}
document.addEventListener("DOMContentLoaded", loadTopAll);

async function loadTopAll(){

  try{
    const res = await fetch("/players/top-all");
    const data = await res.json();

    const batBox = document.getElementById("topBattersList");
    const bowlBox = document.getElementById("topBowlersList");

    batBox.innerHTML = "";
    bowlBox.innerHTML = "";

    data.batters.forEach((p,i)=>{
      batBox.innerHTML += `
        <div class="leader-item">
          <div class="leader-name">${i+1}. ${p.name}</div>
          <div class="leader-value">${p.runs}</div>
        </div>`;
    });

    data.bowlers.forEach((p,i)=>{
      bowlBox.innerHTML += `
        <div class="leader-item">
          <div class="leader-name">${i+1}. ${p.name}</div>
          <div class="leader-value">${p.wickets}</div>
        </div>`;
    });

  }catch(e){
    console.log("Top players load failed");
  }
}


